"""交互式评估菜单。"""
import json
import os
from pathlib import Path

from .utils import DEFAULT_TESTSETS_DIR, DEFAULT_REPORTS_DIR, resolve_path
from .document_loader import load_langchain_docs
from .testset_gen import generate_testset
from .report import print_report


def _list_json_files(directory: str) -> list[Path]:
    """List JSON files in a directory, sorted by modification time (newest first)."""
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def _menu_generate():
    """Interactive test set generation."""
    from config import settings

    print("\n--- 生成测试集 ---")
    size = input("测试用例数量 [50]: ").strip()
    try:
        size = int(size) if size else 50
    except ValueError:
        print("Error: 请输入数字")
        return

    default_output = str(DEFAULT_TESTSETS_DIR / "testset.json")
    output = input(f"输出路径 [{default_output}]: ").strip()
    output = output or default_output

    docs_dir = settings.DOCUMENTS_DIR
    if not os.path.exists(docs_dir):
        print(f"Error: Documents directory not found: {docs_dir}")
        return

    docs = load_langchain_docs(docs_dir)
    if not docs:
        print("Error: No documents loaded")
        return

    print(f"\nLoaded {len(docs)} pages from {docs_dir}")
    print(f"Generating {size} test cases...")

    test_cases = generate_testset(documents=docs, size=size)

    output_path = resolve_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated {len(test_cases)} test cases → {output_path}")
    print("Review and edit the test set before running evaluate.")


def _menu_evaluate():
    """Interactive evaluation."""
    from .cli import cmd_evaluate

    print("\n--- 运行评估 ---")

    testsets = _list_json_files(str(DEFAULT_TESTSETS_DIR))
    if not testsets:
        print(f"Error: No test sets found in {DEFAULT_TESTSETS_DIR}")
        print("Run '生成测试集' first.")
        return

    print("可用的测试集：")
    for i, f in enumerate(testsets, 1):
        print(f"  {i}. {f.name}")
    print()

    choice = input("选择测试集 [1]: ").strip()
    try:
        choice = int(choice) if choice else 1
    except ValueError:
        print("Error: 请输入数字")
        return
    if choice < 1 or choice > len(testsets):
        print("Error: Invalid choice")
        return

    testset_path = testsets[choice - 1]

    threshold_input = input("最低分数线（回车跳过）: ").strip()
    try:
        threshold = float(threshold_input) if threshold_input else None
    except ValueError:
        print("Error: 请输入数字")
        return

    class Args:
        pass

    args = Args()
    args.testset = str(testset_path)
    args.report = None
    args.top_k = None
    args.threshold = threshold
    args.judge_model = None
    args.mode = "single"

    cmd_evaluate(args)


def _menu_reports():
    """View historical reports."""
    reports = _list_json_files(str(DEFAULT_REPORTS_DIR))
    if not reports:
        print("\n暂无评估报告。")
        return

    print("\n--- 历史报告 ---")
    for i, f in enumerate(reports, 1):
        print(f"  {i}. {f.name}")
    print()

    choice = input("查看报告（输入编号，回车返回）: ").strip()
    if not choice:
        return

    try:
        choice = int(choice)
    except ValueError:
        print("Error: 请输入数字")
        return
    if choice < 1 or choice > len(reports):
        print("Error: Invalid choice")
        return

    report_path = reports[choice - 1]
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    print_report(report)


def interactive_menu():
    """Main interactive menu loop."""
    while True:
        print("\n=== RagMate RAGAS 评估工具 ===\n")
        print("  1. 生成测试集")
        print("  2. 运行评估")
        print("  3. 查看历史报告")
        print("  0. 退出")
        print()

        choice = input("请选择 > ").strip()

        if choice == "1":
            _menu_generate()
        elif choice == "2":
            _menu_evaluate()
        elif choice == "3":
            _menu_reports()
        elif choice == "0":
            print("Bye!")
            break
        else:
            print("无效选择，请重试。")
