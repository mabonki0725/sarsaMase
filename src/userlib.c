#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**********************
  ‹¤’Êƒƒ‚ƒŠ[Š„‚è“–‚Ä
***********************/
char *comMalloc(size)
int size;
{
   char *p;
   p=(char *)malloc(size);
   if(p) memset(p,'\0',size); 
   else  printf("malloc0 —Ìˆæ‚ÍŠm•Ûo—ˆ‚Ü‚¹‚ñ!!\n");
   return(p);
}
/********************
  ‹¤’Ê•¶šŠ„‚è“–‚Ä
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
