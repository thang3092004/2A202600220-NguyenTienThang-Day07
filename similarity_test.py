import os
from dotenv import load_dotenv
from src.chunking import compute_similarity
from src.embeddings import OpenAIEmbedder

load_dotenv()

def main():
    e = OpenAIEmbedder()
    pairs = [
        ("Hôm nay trời nắng đẹp.", "Thời tiết hôm nay rất tốt."),
        ("Tôi thích ăn pizza.", "Con mèo đang ngủ trên ghế."),
        ("Suboi là rapper.", "Trang Anh là nghệ sĩ hiphop."),
        ("Hà Nội là thủ đô Việt Nam.", "Paris là thủ đô nước Pháp."),
        ("ICD là quán quân KOR.", "Dế Choắt thắng Rap Việt.")
    ]
    
    for a, b in pairs:
        score = compute_similarity(e(a), e(b))
        print(f"{a} | {b} | {round(score, 4)}")

if __name__ == "__main__":
    main()
