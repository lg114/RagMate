import os


def canonical_source(source: str) -> str:
    """归一化来源文件名，用于检索去重。

    只处理明显的副本/重复标记，不改变文件名中的有意义数字。
    例如：
      'report_副本.pdf' → 'report.pdf'
      'report (1).pdf' → 'report.pdf'
      'data_1.xlsx' → 'data_1.xlsx'  （不改变，因为 1 是有意义的编号）
    """
    if not source:
        return ""
    name = os.path.splitext(source)
    base, ext = name[0], name[1]
    # 去掉 副本、copy 等标记
    for suffix in ["_副本", " (副本)", "_copy", " (copy)"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    # 去掉 (1)、(2) 等括号编号
    if base.endswith(")") and "(" in base:
        idx = base.rfind("(")
        if base[idx + 1 : -1].isdigit():
            base = base[:idx]
    return (base + ext).lower()
