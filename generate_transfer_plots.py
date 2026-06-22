import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

with open('results.pkl', 'rb') as f:
    tl_results = pickle.load(f)
with open('results_custom_cnn_no_augmentation.pkl', 'rb') as f:
    cc_results = pickle.load(f)

ALL_RESULTS = {**tl_results, **cc_results}
MODEL_ORDER = ['vgg', 'custom_cnn', 'resnet18', 'resnet50']
MODEL_LABELS = {'vgg': 'VGG16', 'custom_cnn': 'CustomCNN',
                'resnet18': 'ResNet18', 'resnet50': 'ResNet50'}
CLASS_NAMES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

model_names = [m for m in MODEL_ORDER if m in ALL_RESULTS]
model_labels = [MODEL_LABELS[m] for m in model_names]


def plot_model_comparison():
    accs = [np.mean(ALL_RESULTS[m]['accuracies']) for m in model_names]
    bal_accs = [np.mean(ALL_RESULTS[m]['balanced_accuracies']) for m in model_names]
    precs = [np.mean(ALL_RESULTS[m]['precisions']) for m in model_names]
    recs = [np.mean(ALL_RESULTS[m]['recalls']) for m in model_names]

    x = np.arange(len(model_names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - 1.5*w, accs, w, label='Accuracy', color='#1f77b4')
    bars2 = ax.bar(x - 0.5*w, bal_accs, w, label='Balanced Accuracy', color='#ff7f0e')
    bars3 = ax.bar(x + 0.5*w, precs, w, label='Precision (weighted)', color='#2ca02c')
    bars4 = ax.bar(x + 1.5*w, recs, w, label='Recall (weighted)', color='#d62728')

    for bar in [bars1, bars2, bars3, bars4]:
        for rect in bar:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                        fontsize=9, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=13)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel('Wartość metryki')
    ax.set_title('Porównanie modeli — średnie metryki (10 foldów)', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig('img/model_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("Zapisano img/model_comparison.pdf")


def plot_confusion_matrices():
    n = len(model_names)
    cols = min(4, n)
    rows = (n + cols - 1) // cols if n > 2 else 1
    if n <= 2:
        rows, cols = 1, n

    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 6.5*rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flat

    for ax, m in zip(axes, model_names):
        mean_cm = np.mean(ALL_RESULTS[m]['confusion_matrices'], axis=0)
        mean_cm_norm = mean_cm / mean_cm.sum(axis=1, keepdims=True)
        bal_acc = np.mean(ALL_RESULTS[m]['balanced_accuracies'])

        sns.heatmap(mean_cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    vmin=0, vmax=1, ax=ax, cbar=True,
                    annot_kws={'fontsize': 10})
        ax.set_title(f'{MODEL_LABELS[m]}\nBalanced Accuracy: {bal_acc:.4f}',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Predykcja')
        ax.set_ylabel('Rzeczywista klasa')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig('img/confusion_matrices.pdf', bbox_inches='tight')
    plt.close()
    print("Zapisano img/confusion_matrices.pdf")


def plot_per_class_metrics():
    n = len(model_names)
    if n == 4:
        rows, cols = 2, 2
    elif n <= 2:
        rows, cols = 1, n
    else:
        cols = min(4, n)
        rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 5.5*rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flat

    for ax, m in zip(axes, model_names):
        mean_cm = np.mean(ALL_RESULTS[m]['confusion_matrices'], axis=0)
        recall = np.diag(mean_cm) / (mean_cm.sum(axis=1) + 1e-8)
        precision = np.diag(mean_cm) / (mean_cm.sum(axis=0) + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        x = np.arange(len(CLASS_NAMES))
        w = 0.25
        ax.bar(x - w, precision, w, label='Precision', color='#2ca02c')
        ax.bar(x, recall, w, label='Recall', color='#1f77b4')
        ax.bar(x + w, f1, w, label='F1', color='#d62728')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(f'Per-class metrics: {MODEL_LABELS[m]}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig('img/per_class_metrics.pdf', bbox_inches='tight')
    plt.close()
    print("Zapisano img/per_class_metrics.pdf")


def print_error_analysis():
    for m in model_names:
        mean_cm = np.mean(ALL_RESULTS[m]['confusion_matrices'], axis=0)
        mean_cm_norm = mean_cm / (mean_cm.sum(axis=1, keepdims=True) + 1e-8)

        errors = [(mean_cm_norm[i, j], CLASS_NAMES[i], CLASS_NAMES[j])
                  for i in range(len(CLASS_NAMES))
                  for j in range(len(CLASS_NAMES))
                  if i != j]
        errors.sort(reverse=True)

        print(f"\n=== {MODEL_LABELS[m]} — najczęstsze błędy klasyfikacji ===")
        print(f"{'Prawdziwa':<12} → {'Predykcja':<12}  {'Częstość':>8}")
        for rate, true_cls, pred_cls in errors[:5]:
            print(f"{true_cls:<12} → {pred_cls:<12}  {rate:>8.2%}")


def fmt_val(mean, std):
    """Format jako średnia(std), gdzie std = int(round(std*10000))."""
    s = int(round(std * 10000))
    return f"{mean:.4f}({s:0>2d})"


def print_summary_table():
    print("\n=== Tabela wyników (średnia(10000×std)) ===")
    header = f"{'Model':<12} {'Accuracy':>16} {'Bal.Acc':>16} {'Precision':>16} {'Recall':>16}"
    print(header)
    print("-" * len(header))
    for m in model_names:
        d = ALL_RESULTS[m]
        a_m, a_s = np.mean(d['accuracies']), np.std(d['accuracies'], ddof=1)
        b_m, b_s = np.mean(d['balanced_accuracies']), np.std(d['balanced_accuracies'], ddof=1)
        p_m, p_s = np.mean(d['precisions']), np.std(d['precisions'], ddof=1)
        r_m, r_s = np.mean(d['recalls']), np.std(d['recalls'], ddof=1)
        print(f"{MODEL_LABELS[m]:<12} {fmt_val(a_m, a_s):>16} {fmt_val(b_m, b_s):>16} "
              f"{fmt_val(p_m, p_s):>16} {fmt_val(r_m, r_s):>16}")


if __name__ == '__main__':
    import sys

    plot_model_comparison()
    plot_confusion_matrices()
    plot_per_class_metrics()

    orig_stdout = sys.stdout
    with open("out.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        print("# Wyniki eksperymentow - emotion detection FER")
        print(f"# Modele: {', '.join(model_labels)}\n")
        print_error_analysis()
        print_summary_table()
    sys.stdout = orig_stdout

    print("[Wyniki zapisano do out.txt]")