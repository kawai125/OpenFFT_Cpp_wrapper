/*******************************************************
*    This is the color printer for terminal (using escape secuence).
******************************************************/

#pragma once

#include <cstdio>
#include <string>
#include <iostream>

void print_green(const std::string &str){
    //--- green output
    printf( ("\033[32m" + str + "\033[m").c_str() );
}
void print_yellow(const std::string &str){
    //--- green output
    printf( ("\033[33m" + str + "\033[m").c_str() );
}
void print_red(const std::string &str){
    //--- red output
    printf( ("\033[31m" + str + "\033[m").c_str() );
}

template <class T>
void oss_green(std::ostream& s, const T &out){
    s << "\033[32m" << out << "\033[m";
}
template <class T>
void oss_yellow(std::ostream& s, const T &out){
    s << "\033[33m" << out << "\033[m";
}
template <class T>
void oss_red(std::ostream& s, const T &out){
    s << "\033[31m" << out << "\033[m";
}
