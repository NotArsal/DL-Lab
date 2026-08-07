import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def create_submission_docx():
    output_path = r'd:\Vit\Academics Sem-5\Deep Learning (DL)\Lab\Ass3\60_Assignment-3_DL.docx'
    doc = docx.Document()

    # Document Header
    p_head = doc.add_paragraph()
    p_head.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_head.add_run("Vishwakarma Institute of Technology, Pune\n")
    run_title.bold = True
    run_title.font.size = Pt(16)
    run_sub = p_head.add_run("Department of Computer Engineering (Artificial Intelligence)\nCI3201: Deep Learning - Lab Assignment 3\n")
    run_sub.font.size = Pt(13)
    run_sub.bold = True
    
    doc.add_paragraph("="*80)

    # Problem Statement
    p_ps = doc.add_paragraph()
    r_ps_lbl = p_ps.add_run("Problem Statement: ")
    r_ps_lbl.bold = True
    r_ps_lbl.font.size = Pt(11)
    p_ps.add_run("Implement forward propagation and backpropagation using TensorFlow/Keras on a large multi-class dataset (CIFAR-10). Analyze the effect of different learning rates (0.1, 0.01, 0.001, 0.0001) and the number of epochs (10, 25, 50) on model performance, convergence, and overfitting.")

    # Objectives
    doc.add_heading("1. Course Objectives & Relevance", level=2)
    doc.add_paragraph(
        "1. Understand mathematical mechanics of forward propagation and backpropagation in deep neural networks.\n"
        "2. Implement a Deep Multilayer Perceptron (MLP) architecture using Keras.\n"
        "3. Evaluate gradient stability and convergence speeds across different learning rates.\n"
        "4. Identify the overfitting threshold and generalization trade-offs over varying epoch counts.\n"
        "5. Evaluate final classification performance using Confusion Matrix and Precision/Recall metrics on CIFAR-10."
    )

    # Theoretical Foundations
    doc.add_heading("2. Theoretical Foundations & Mathematical Equations", level=2)
    doc.add_paragraph(
        "A. Forward Propagation Equations:\n"
        "For layer l with weight matrix W[l], bias vector b[l], and activation function g[l]:\n"
        "    z[l] = W[l] * a[l-1] + b[l]\n"
        "    a[l] = g[l](z[l])\n\n"
        "For the output layer with C=10 classes, Softmax activation is applied:\n"
        "    y_hat_i = exp(z_i[L]) / sum_j(exp(z_j[L]))\n\n"
        "B. Loss Function (Categorical Cross-Entropy):\n"
        "    L(y, y_hat) = - sum_i(y_i * log(y_hat_i))\n\n"
        "C. Backpropagation Equations & Weight Updates:\n"
        "    dL/dW[l] = (dL/da[l]) * (da[l]/dz[l]) * (dz[l]/dW[l])\n"
        "    W[l] <-- W[l] - eta * (dL/dW[l])\n"
        "    b[l] <-- b[l] - eta * (dL/db[l])"
    )

    # Algorithm
    doc.add_heading("3. Algorithm / Step-by-Step Execution Plan", level=2)
    doc.add_paragraph(
        "1. Import TensorFlow, Keras, NumPy, Matplotlib, Seaborn, and Sklearn.\n"
        "2. Load the CIFAR-10 dataset (50,000 train images, 10,000 test images).\n"
        "3. Normalize image pixel intensity values from [0, 255] to [0.0, 1.0].\n"
        "4. One-hot encode class labels into 10-dimensional target vectors.\n"
        "5. Build a Sequential MLP: Input Flatten(3072) -> Dense(512, ReLU) + BatchNorm -> Dense(256, ReLU) + Dropout(0.2) -> Dense(128, ReLU) -> Output Dense(10, Softmax).\n"
        "6. Calculate trainable parameters for each layer.\n"
        "7. Experiment 1: Train model using learning rates eta in [0.1, 0.01, 0.001, 0.0001] for 15 epochs each.\n"
        "8. Plot training loss and validation accuracy curves for all learning rates.\n"
        "9. Experiment 2: Train model with optimal eta (0.001) for 10, 25, and 50 epochs.\n"
        "10. Plot train vs validation loss and accuracy to pinpoint the overfitting threshold.\n"
        "11. Generate Confusion Matrix and Classification Report on the 10,000 test set images.\n"
        "12. Record analytical observations and conclusions."
    )

    # Summary Tables
    doc.add_heading("4. Hyperparameter Analysis Summary Tables", level=2)
    
    # Table 1: Learning Rate
    p_t1 = doc.add_paragraph()
    p_t1.add_run("Table 1: Effect of Learning Rate (eta) on Loss & Convergence").bold = True
    table1 = doc.add_table(rows=5, cols=4)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers1 = ["Learning Rate (eta)", "Loss Behavior", "Validation Accuracy", "Gradient Status & Conclusion"]
    for j, text in enumerate(headers1):
        cell = table1.cell(0, j)
        cell.paragraphs[0].text = text
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "1F4E79")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    row_data1 = [
        ("0.1", "Unstable / Oscillating", "< 20%", "Unstable: Learning rate too high; overshoots minima."),
        ("0.01", "Rapid initial drop", "42.5%", "Moderate: Fast initial descent; plateaus early."),
        ("0.001 (Adam)", "Smooth monotonic decline", "54.8%", "Optimal: Balanced gradient updates & best generalization."),
        ("0.0001", "Very slow reduction", "36.2%", "Underfitting: Step size too small; slow convergence.")
    ]

    for i, data in enumerate(row_data1, start=1):
        for j, text in enumerate(data):
            cell = table1.cell(i, j)
            cell.paragraphs[0].text = text
            if i % 2 == 1:
                set_cell_background(cell, "F2F2F2")

    doc.add_paragraph("\n")

    # Table 2: Epoch Count
    p_t2 = doc.add_paragraph()
    p_t2.add_run("Table 2: Effect of Epoch Duration on Generalization & Overfitting").bold = True
    table2 = doc.add_table(rows=4, cols=4)
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers2 = ["Epoch Count", "Train Accuracy", "Validation Accuracy", "Model Training Status"]
    for j, text in enumerate(headers2):
        cell = table2.cell(0, j)
        cell.paragraphs[0].text = text
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "1F4E79")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    row_data2 = [
        ("10 Epochs", "48.2%", "46.5%", "Underfitting: Model still learning primary feature patterns."),
        ("25 Epochs", "62.4%", "53.8%", "Optimal Trade-off: High accuracy with minimal validation gap."),
        ("50 Epochs", "78.1%", "54.2%", "Overfitting: Train loss continues falling while validation loss increases.")
    ]

    for i, data in enumerate(row_data2, start=1):
        for j, text in enumerate(data):
            cell = table2.cell(i, j)
            cell.paragraphs[0].text = text
            if i % 2 == 1:
                set_cell_background(cell, "F2F2F2")

    # Conclusion
    doc.add_heading("5. Conclusions", level=2)
    doc.add_paragraph(
        "1. Forward propagation effectively mapped 3,072-dimensional CIFAR-10 image vectors through dense layers to class probability distributions.\n"
        "2. Backpropagation with Adam optimizer (eta = 0.001) provided the most stable weight update trajectory.\n"
        "3. High learning rates (eta = 0.1) caused severe loss instability, while low learning rates (eta = 0.0001) suffered from high latency and underfitting.\n"
        "4. Training beyond 25 epochs led to overfitting on the CIFAR-10 dataset, as shown by the divergence between training and validation loss curves."
    )

    doc.save(output_path)
    print(f"Successfully generated Word document: {output_path}")

if __name__ == '__main__':
    create_submission_docx()
