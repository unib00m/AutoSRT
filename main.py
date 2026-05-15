import stable_whisper
import re
import os
import sys
import multiprocessing

def format_timestamp(seconds: float):
    """將秒數轉換為 SRT 標準時間格式 (HH:MM:SS,mmm)"""
    ms = int((seconds % 1) * 1000)
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def auto_generate_srt(audio_path: str, raw_text_path: str, output_srt_path: str):
    """
    回歸最初版：不加上任何自訂的排版、時間微調與字元比對邏輯。
    完全讓 stable-ts 的底層原生系統來決定時間軸與分段。
    """
    
    print("步驟 1/4：讀取逐字稿...")
    with open(raw_text_path, 'r', encoding='utf-8') as f:
        original_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # 使用特殊符號作為強制斷句的錨點，避免換行符號被 AI 忽略
    separator = '█'
    transcript_text = separator.join(original_lines) + separator
    
    print("步驟 2/4：載入模型...")
    # 升級為 'small' 模型：雖然慢一點點，但能大幅降低 AI 在無聲區間產生「幻覺」的機率
    model = stable_whisper.load_model('small')
    
    print("步驟 3/4：強制對齊...")
    result = model.align(audio_path, transcript_text, language='zh', vad=True)
    
    print("步驟 4/4：輸出 SRT...")
    # 直接攤平 AI 聽到的所有單字，我們手動來組合 SRT！
    flat_words = [word for segment in result.segments for word in segment.words]
    srt_content = ""
    line_idx = 0
    current_start = None
    current_end = None
    last_end_time = 0.0
    buffered_sentence = None  # 建立一個暫存區，用來實現「橋接貼齊」
    
    for w in flat_words:
        if current_start is None:
            # 統一讓起點微幅提早 0.05 秒，但絕對不能早於上一句的結束時間，避免重疊！
            current_start = max(last_end_time + 0.001, w.start - 0.05)
            
        clean_word = w.word.replace(separator, '').strip()
        has_speech = any(c.isalnum() for c in clean_word)
        
        # 【終極抓鬼邏輯：只採納真正發音字元的結束時間】
        # 如果是純標點或錨點符號，絕對不更新 current_end，避免被 AI 拖到無聲區間！
        if has_speech:
            # 單字級別緊箍咒：防止最後一個字本身被拉長 (每個字最多給 0.4秒 + 0.6秒彈性)
            max_word_dur = len(clean_word) * 0.4 + 0.6
            current_end = min(w.end, w.start + max_word_dur)
        elif current_end is None:
            # 防呆：如果一開始只有純符號
            current_end = w.end
        
        # 只要在這個字裡面看到我們埋的錨點，就強制結算輸出你原本的這一行！
        for _ in range(w.word.count(separator)):
            if line_idx < len(original_lines):
                # 防呆：確保在極端狀況下有起始時間且結束時間大於開始時間
                if current_start is None:
                    current_start = last_end_time + 0.001
                if current_end <= current_start:
                    current_end = current_start + 0.1
                    
                if buffered_sentence is not None:
                    # 【防閃爍：字幕橋接機制】如果兩句話間隔小於 0.8 秒，讓前一句的結尾直接貼齊下一句的開頭
                    gap = current_start - buffered_sentence['end']
                    if 0 <= gap <= 0.8:
                        buffered_sentence['end'] = current_start - 0.001
                    
                    # 【防閃爍：最短停留時間】確保極短的語氣詞 (如 Hoo!) 不會一閃而過
                    # 設定字幕至少停留 0.6 秒，但最高極限絕對不能蓋到下一句話！
                    if (buffered_sentence['end'] - buffered_sentence['start']) < 0.6:
                        buffered_sentence['end'] = min(buffered_sentence['start'] + 0.6, current_start - 0.001)
                    
                    srt_content += f"{buffered_sentence['idx']}\n{format_timestamp(buffered_sentence['start'])} --> {format_timestamp(buffered_sentence['end'])}\n{buffered_sentence['text']}\n\n"
                    
                # 把這句話放入暫存區，等下一句話確認時間後再決定要不要延長它
                buffered_sentence = {
                    'idx': line_idx + 1,
                    'start': current_start,
                    'end': current_end,
                    'text': original_lines[line_idx]
                }
                
                line_idx += 1
                last_end_time = current_end
            # 清空起點，準備迎接下一句
            current_start = None
            
    # 防呆機制：如果迴圈跑完，但你的文字檔還有剩餘沒被輸出的句子，全部強制印出
    while line_idx < len(original_lines):
        if current_start is None:
            current_start = max(last_end_time + 0.001, current_end if current_end else 0.0)
        current_end = current_start + 0.1
        
        if buffered_sentence is not None:
            gap = current_start - buffered_sentence['end']
            if 0 <= gap <= 0.8:
                buffered_sentence['end'] = current_start - 0.001
                
            # 【防閃爍：最短停留時間】
            if (buffered_sentence['end'] - buffered_sentence['start']) < 0.6:
                buffered_sentence['end'] = min(buffered_sentence['start'] + 0.6, current_start - 0.001)
                
            srt_content += f"{buffered_sentence['idx']}\n{format_timestamp(buffered_sentence['start'])} --> {format_timestamp(buffered_sentence['end'])}\n{buffered_sentence['text']}\n\n"
            
        buffered_sentence = {
            'idx': line_idx + 1,
            'start': current_start,
            'end': current_end,
            'text': original_lines[line_idx]
        }
        line_idx += 1
        last_end_time = current_end
        current_start = None
        
    # 迴圈結束後，把暫存區裡的最後一句話印出來
    if buffered_sentence is not None:
        # 最後一句話後面沒人了，直接安心延長至少 0.6 秒
        if (buffered_sentence['end'] - buffered_sentence['start']) < 0.6:
            buffered_sentence['end'] = buffered_sentence['start'] + 0.6
        srt_content += f"{buffered_sentence['idx']}\n{format_timestamp(buffered_sentence['start'])} --> {format_timestamp(buffered_sentence['end'])}\n{buffered_sentence['text']}\n\n"

    with open(output_srt_path, 'w', encoding='utf-8') as f:
        f.write(srt_content.strip())
        
    print(f"🎉 任務完成！SRT 已順利儲存至：{output_srt_path}")

if __name__ == "__main__":
    # 解決打包後，AI 模型呼叫多執行緒導致程式無限自我複製（Fork Bomb）的問題
    multiprocessing.freeze_support()

    # 自動偵測當前「執行檔」或「腳本」所在的資料夾
    if getattr(sys, 'frozen', False):
        current_dir = os.path.dirname(sys.executable)
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
    all_files = os.listdir(current_dir)
    
    # 支援的影音格式 (未來如果想支援 .wav 或 .mov，直接加在這裡面就可以了！)
    SUPPORTED_MEDIA = ('.mp3', '.mp4', '.wav')
    
    media_files_list = [f for f in all_files if f.lower().endswith(SUPPORTED_MEDIA)]
    txt_files_list = [f for f in all_files if f.lower().endswith('.txt')]
    
    # 【全新智能配對機制】
    # 情況 A：如果資料夾剛好只有 1 個影音檔和 1 個文字檔，直接盲配對，無視檔名！
    if len(media_files_list) == 1 and len(txt_files_list) == 1:
        media_file = media_files_list[0]
        txt_file = txt_files_list[0]
        base_name = os.path.splitext(media_file)[0] # 使用影音檔的名稱作為輸出的 SRT 名稱
        
        print(f"🔍 智能模式：偵測到資料夾內只有一組檔案，將無視檔名自動配對！\n- 影音檔：{media_file}\n- 逐字稿：{txt_file}\n")
        
        AUDIO_FILE = os.path.join(current_dir, media_file)
        TRANSCRIPT_FILE = os.path.join(current_dir, txt_file)
        OUTPUT_SRT = os.path.join(current_dir, f"{base_name}.srt")
        
        auto_generate_srt(AUDIO_FILE, TRANSCRIPT_FILE, OUTPUT_SRT)
        print("✅ 處理完畢！")
        
    else:
        # 情況 B：如果有多個檔案，退回原本的「主檔名嚴格配對模式」
        media_files_dict = {os.path.splitext(f)[0]: f for f in media_files_list}
        txt_files_dict = {os.path.splitext(f)[0]: f for f in txt_files_list}
        matched_names = set(media_files_dict.keys()).intersection(set(txt_files_dict.keys()))
        
        if not matched_names:
            print("⚠️ 找不到可以配對的檔案！")
            print("提示：若資料夾有多個檔案，主檔名必須完全一致。或者您可以每次只放「1部影片+1份txt」，系統將會無視檔名為您自動配對。")
        else:
            print(f"🔍 偵測到 {len(matched_names)} 組同名配對檔案，準備開始批次處理...\n")
            for base_name in matched_names:
                print(f"========== 正在處理: {base_name} ==========")
                AUDIO_FILE = os.path.join(current_dir, media_files_dict[base_name])
                TRANSCRIPT_FILE = os.path.join(current_dir, txt_files_dict[base_name])
                OUTPUT_SRT = os.path.join(current_dir, f"{base_name}.srt")
                auto_generate_srt(AUDIO_FILE, TRANSCRIPT_FILE, OUTPUT_SRT)
                print("===========================================\n")
            print("✅ 所有檔案皆已批次處理完畢！")
