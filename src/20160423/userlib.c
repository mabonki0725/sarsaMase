#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**********************
  ���ʃ������[���蓖��
***********************/
char *comMalloc(size)
int size;
{
   char *p;
   p=(char *)malloc(size);
   if(p) memset(p,'\0',size); 
   else  printf("malloc0 �̈�͊m�ۏo���܂���!!\n");
   return(p);
}
/********************
  ���ʕ������蓖��
*********************/
char *comAssign(adr,str)
char **adr;
char *str;
{
    int len;

    len=strlen(str);
    if(!len) return(NULL);

    *adr=(char *)malloc(len+1);
    memset(*adr,'\0',len+1);
    strcpy(*adr,str);
    return(*adr);
}
