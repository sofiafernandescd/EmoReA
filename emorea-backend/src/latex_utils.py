def classification_report_to_latex(report, dataset_name):
    """Convert a classification report dictionary to a LaTeX table format.
    Args:
        report (dict): Classification report dictionary from sklearn.metrics.classification_report.
        dataset_name (str): Name of the dataset for the table caption.
    Returns:
        str: LaTeX formatted string for the classification report table.
    """
    lines = [
        "\\section*{" + dataset_name + " - Classification Report}",
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Classification metrics for " + dataset_name + " dataset}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Emotion} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-score} & \\textbf{Support} \\\\",
        "\\midrule"
    ]

    for label, metrics in report.items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue
        p = f"{metrics['precision']:.2f}"
        r = f"{metrics['recall']:.2f}"
        f1 = f"{metrics['f1-score']:.2f}"
        s = f"{int(metrics['support'])}"
        lines.append(f"{label.title()} & {p} & {r} & {f1} & {s} \\\\")

    lines.append("\\midrule")
    lines.append(f"\\textbf{{Accuracy}} & \\multicolumn{{4}}{{c}}{{\\textbf{{{report['accuracy']:.2f}}}}} \\\\")
    for avg in ["macro avg", "weighted avg"]:
        m = report[avg]
        p = f"{m['precision']:.2f}"
        r = f"{m['recall']:.2f}"
        f1 = f"{m['f1-score']:.2f}"
        s = f"{int(m['support'])}"
        lines.append(f"\\textbf{{{avg.title()}}} & {p} & {r} & {f1} & {s} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_summary_table(results_dict):
    """Generate a LaTeX table summarizing SVM accuracy results across different feature sets and datasets.
       results = {
            "EmoDB": {"svm_geomaps": 0.843, "svm_fs_geomaps": 0.861, "svm_librosa": 0.817, "svm_fs_librosa": 0.840},
            ...
        }

    Args:
        results_dict (Dict): Dictionary containing SVM accuracy results for different datasets and feature sets.

    Returns:
        str: LaTeX formatted string for the summary table.
    """
    header = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Comparison of SVM accuracy across feature sets and datasets}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{SVM (GeMAPS)} & \\textbf{SVM + FS (GeMAPS)} & \\textbf{SVM (Librosa)} & \\textbf{SVM + FS (Librosa)} \\\\",
        "\\midrule"
    ]
    body = []
    for dataset, scores in results_dict.items():
        row = f"{dataset} & " + " & ".join(f"{100 * scores[k]:.1f}\\%" for k in [
            "svm_geomaps", "svm_fs_geomaps", "svm_librosa", "svm_fs_librosa"]) + " \\\\"
        body.append(row)
    footer = ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(header + body + footer)
