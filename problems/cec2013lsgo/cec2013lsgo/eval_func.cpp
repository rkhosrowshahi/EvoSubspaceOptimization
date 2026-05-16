#include "Header.h"
#include <cassert>

static Benchmarks* bench=NULL;

Benchmarks* generateFuncObj(int funcID);

void set_func(int funcID) {
    bench = generateFuncObj(funcID);
    bench->nextRun();
}

void next_run() {
    bench->nextRun();
}

double eval_sol(double *x) {
    return bench->compute(x);
}

void set_data_dir(char *new_data_dir) {
    string data_dir = new_data_dir;
    bench->set_data_dir(data_dir);
}

void free_func(void) {
   if (bench) {
      delete bench;
   }

   bench = NULL;
}

// create new object of class with default setting
Benchmarks* generateFuncObj(int funcID){
  Benchmarks *fp;
  // run each of specified function in "configure.ini"
  if (funcID==1){
    fp = new F1();
  }else if (funcID==2){
    fp = new F2();
  }else if (funcID==3){
    fp = new F3();
  }else if (funcID==4){
    fp = new F4();
  }else if (funcID==5){
    fp = new F5();
  }else if (funcID==6){
    fp = new F6();
  }else if (funcID==7){
    fp = new F7();
  }else if (funcID==8){
    fp = new F8();
  }else if (funcID==9){
    fp = new F9();
  }else if (funcID==10){
    fp = new F10();
  }else if (funcID==11){
    fp = new F11();
  }else if (funcID==12){
    fp = new F12();
  }else if (funcID==13){
    fp = new F13();
  }else if (funcID==14){
    fp = new F14();
  }else if (funcID==15){
    fp = new F15();
  }else if (funcID==16){
    fp = new F16();
  }else if (funcID==17){
    fp = new F17();
  }else if (funcID==18){
    fp = new F18();
  }else if (funcID==19){
    fp = new F19();
  }else if (funcID==20){
    fp = new F20();
  }else if (funcID==21){
    fp = new F21();
  }else if (funcID==22){
    fp = new F22();
  }else if (funcID==23){
    fp = new F23();
  }else if (funcID==24){
    fp = new F24();
  }else if (funcID==25){
    fp = new F25();
  }else{
    cerr<<"Fail to locate Specified Function Index"<<endl;
    exit(-1);
  }
  return fp;
}
