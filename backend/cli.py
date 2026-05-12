"""RagMate CLI - 简单的命令行界面"""

import asyncio
import sys

from database import init_db
from eval.runner import run_eval
from ingest import ingest_documents
from retriever import retrieve
from chat import chat


def safe_print(text: str):
    """安全打印，处理 Windows GBK 编码问题"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding, errors="replace"))


def menu():
    print("\n=== RagMate CLI ===")
    print("1. 摄入文档 (ingest documents)")
    print("2. 检索文档 (retrieve documents)")
    print("3. 聊天 (chat)")
    print("4. 评估 (evaluate)")
    print("5. 退出 (exit)")
    print("==================")


INGEST_TIMEOUT = 3600  # 入库超时（秒）


def handle_ingest():
    print("正在初始化数据库...")
    try:
        asyncio.run(init_db())
    except Exception as e:
        safe_print(f"数据库初始化失败: {e}")
        return
    print("正在摄入文档...")
    try:
        asyncio.run(asyncio.wait_for(asyncio.to_thread(ingest_documents), timeout=INGEST_TIMEOUT))
        print("完成！")
    except asyncio.TimeoutError:
        safe_print(f"摄入超时（{INGEST_TIMEOUT}s），请重试或减少文档数量")
    except Exception as e:
        safe_print(f"摄入失败: {e}")


def handle_retrieve():
    query = input("输入查询: ").strip()
    if not query:
        print("查询不能为空")
        return
    try:
        results = asyncio.run(asyncio.wait_for(
            asyncio.to_thread(retrieve, query, k=5),
            timeout=30,
        ))
        safe_print(f"\n找到 {len(results)} 条结果:\n")
        for i, r in enumerate(results, 1):
            source = r.get("source", "")
            page = r.get("page")
            loc = f"【{source}】" + (f" 第{page + 1}页" if page is not None else "")
            safe_print(f"[{i}] {loc} {r['text'][:200]}...")
    except asyncio.TimeoutError:
        safe_print("检索超时（30s），请重试")
    except Exception as e:
        safe_print(f"检索失败: {e}")
        safe_print("请确认 Milvus 服务是否已启动 (docker start milvus-standalone)")


CHAT_TIMEOUT = 180  # 聊天超时（秒）


async def handle_chat():
    print("输入你的问题（输入 q 退出）:")
    while True:
        query = input("\n你: ").strip()
        if query.lower() == 'q':
            break
        if not query:
            continue

        safe_print("思考中...")
        try:
            response = await asyncio.wait_for(chat(query), timeout=CHAT_TIMEOUT)
            safe_print(f"\nRagMate: {response['response']}")
        except asyncio.TimeoutError:
            safe_print("处理超时，请稍后重试")
        except Exception as e:
            safe_print(f"错误: {e}")
            safe_print("请确认:\n"
                        "  1. Milvus 服务是否已启动\n"
                        "  2. LLM API 配置是否正确")


def handle_eval():
    try:
        run_eval()
    except Exception as e:
        safe_print(f"评估失败: {e}")


def main():
    print("欢迎使用 RagMate CLI")

    while True:
        menu()
        choice = input("选择 (1-5): ").strip()

        if choice == '1':
            handle_ingest()
        elif choice == '2':
            handle_retrieve()
        elif choice == '3':
            asyncio.run(handle_chat())
        elif choice == '4':
            handle_eval()
        elif choice == '5':
            print("再见!")
            sys.exit(0)
        else:
            print("无效选择，请重试")


if __name__ == "__main__":
    main()
