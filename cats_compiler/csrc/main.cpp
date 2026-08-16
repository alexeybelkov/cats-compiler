#include <iostream>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>

int main() {
  llvm::LLVMContext context;
  llvm::Module module("cats_compiler", context);
  llvm::IRBuilder builder(context);
  return 0;
}