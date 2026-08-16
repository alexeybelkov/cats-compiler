#include "lexer.hpp"
#include <string>

enum Token {
    teof = -1,
    tdef = -2,
    textern = -3,
    tidentifier = -4,
    tnumber = -5,
};

static std::string IdentifierStr;
static double NumVal;

static int gettok() {
    static int LastChar = ' ';
    while (isspace(LastChar)) LastChar = getchar();
}