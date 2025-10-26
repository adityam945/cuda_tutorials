#include "mainwindow.h"
#include <QApplication>
extern "C" void runCudaAdd();

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    runCudaAdd();  // Run CUDA demo
    return a.exec();
}
