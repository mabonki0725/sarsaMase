#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**********************
  共通メモリー割り当て
***********************/
char *comMalloc(size)
int size;
{
   char *p;
   p=(char *)malloc(size);
   if(p) memset(p,'\0',size); 
   else  printf("malloc0 領域は確保出来ません!!\n");
   return(p);
}
/********************
  共通文字割り当て
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
