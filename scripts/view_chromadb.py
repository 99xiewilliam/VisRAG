#!/usr/bin/env python3
"""
简单的 ChromaDB 查看脚本
用法: python scripts/view_chromadb.py
"""
import chromadb
import json
from pathlib import Path

def main():
    # 从配置读取路径
    db_path = Path(__file__).parent.parent / "output" / "chroma_db"
    
    print("=" * 80)
    print(f"ChromaDB 查看器 - {db_path}")
    print("=" * 80)
    
    client = chromadb.PersistentClient(path=str(db_path))
    collections = client.list_collections()
    
    if not collections:
        print("\n❌ 没有找到任何 collection")
        return
    
    print(f"\n📊 找到 {len(collections)} 个 collection:\n")
    
    for i, col in enumerate(collections, 1):
        count = col.count()
        print(f"{i}. Collection: {col.name}")
        print(f"   ID: {col.id}")
        print(f"   向量数量: {count}")
        
        # 获取样本数据
        if count > 0:
            try:
                sample = col.peek(limit=min(3, count))
                print(f"   样本 ID: {sample['ids'][:3]}")
                
                if sample.get('metadatas') and sample['metadatas']:
                    print(f"   元数据示例:")
                    for idx, meta in enumerate(sample['metadatas'][:2], 1):
                        if meta:
                            print(f"     [{idx}] {json.dumps(meta, ensure_ascii=False, indent=6)}")
                
                if sample.get('documents') and sample['documents']:
                    print(f"   文档示例:")
                    for idx, doc in enumerate(sample['documents'][:2], 1):
                        if doc:
                            preview = doc[:100] + "..." if len(doc) > 100 else doc
                            print(f"     [{idx}] {preview}")
            except Exception as e:
                print(f"   ⚠️  获取样本失败: {e}")
        
        print()
    
    print("=" * 80)

if __name__ == "__main__":
    main()
