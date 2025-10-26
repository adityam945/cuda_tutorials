#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>

// External CUDA function
extern "C" void launchVectorAdd(const float* h_A, const float* h_B, float* h_C, int N);

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->runCudaButton, &QPushButton::clicked, this, &MainWindow::on_runCudaButton_clicked);
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::on_runCudaButton_clicked() {
    const int N = 10;
    float A[N], B[N], C[N];

    for (int i = 0; i < N; ++i) {
        A[i] = i;
        B[i] = i * 10;
    }

    launchVectorAdd(A, B, C, N);

    QString result;
    for (int i = 0; i < N; ++i)
        result += QString("%1 + %2 = %3\n").arg(A[i]).arg(B[i]).arg(C[i]);

    ui->textEdit->setText(result);
}
