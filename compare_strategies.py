from src.chunking import ChunkingStrategyComparator
from pathlib import Path

def run_comparison():
    comparator = ChunkingStrategyComparator()
    # Chọn 2 tài liệu tiêu biểu
    docs = ["suboi.md", "minh_lai.md"]
    data_dir = Path("data/raw_data")
    
    # Thiết lập tham số size mẫu
    CHUNK_SIZE = 500
    
    print(f"{'File':<15} | {'Strategy':<15} | {'Count':<6} | {'Avg Len':<8}")
    print("-" * 50)
    
    for doc_name in docs:
        file_path = data_dir / doc_name
        if not file_path.exists():
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        results = comparator.compare(text, chunk_size=CHUNK_SIZE)
        
        for strategy, stats in results.items():
            print(f"{doc_name:<15} | {strategy:<15} | {stats['count']:<6} | {round(stats['avg_length'], 1):<8}")
        print("-" * 50)

if __name__ == "__main__":
    run_comparison()
