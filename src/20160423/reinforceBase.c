/*
This is an example program for reinforcement learning with linear 
function approximation.  The code follows the psuedo-code for linear, 
gradient-descent Sarsa(lambda) given in Figure 8.8 of the book 
"Reinforcement Learning: An Introduction", by Sutton and Barto.

This version is kept simple, at the cost of efficiency.  
Eligibility traces are implemented naively.  Features sets are arrays.

Before running the program you need to load the tile-coding 
software, available at http://envy.cs.umass.edu/~rich/tiles.C and tiles.h
(see http://envy.cs.umass.edu/~rich/tiles.html for documentation).

The code below is in two main parts: 1) General RL code, and
2) Mountain Car code.

Written by Rich Sutton 12/17/00
 */

//#include <iostream>
#include <stdio.h>
#include "tile.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "userlib.h"

typedef int bool;

#define N 4000                         // number of parameters to theta, memory size
#define M 4                            // number of actions
#define NUM_TILINGS 10                 // number of tilings in tile coding

#define ON  1
#define OFF 0

#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

// Global RL variables:
double Q[M];                           // the action values
double theta[N];                       // modifyable parameter vector, aka memory, weights
double e[N];                           // eligibility traces
int F[M][NUM_TILINGS];                 // sets of features, one for each action
    
// Standard RL parameters:
#define epsilon 0.0                    // probability of random action
#define alpha 0.5                      // step size parameter
#define lambda 0.9                     // trace-decay parameters
#define gamma 1                        // discount-rate parameters

// Profiles:
int episode(int max_steps,FILE *fw,double **,int,int);   // do one episode, return length
void load_Qall();                      // compute action values for current theta, F
void load_Q(int a);                    // compute one action value for current theta, F
int argmax(double Q[M]);               // compute argmax action from Q
bool with_probability(double p);       // helper - true with given probability
void load_F();                         // compute feature sets for current state
void mcar_init();                      // initialize car state
double  mcar_step(int a,int no,FILE *fw,double **,int,int); // update car state for given action
bool mcar_goal_p (double **,int,int);  // is car at goal?
double  goal_dist();

static int posX,posY;
static int goalX,goalY;
// The main program just does a bunch or runs, each consisting of some episodes.
// It prints out the length (number of steps) of each episode.
int main(int argc,char *argv[]) 
{

  int run;
  int episode_num;
  int ret;
  FILE *fp,*fw;
  char record[2048];
  char *pc;
  int i,j,m,n;
  double **data;
  int nSize,mSize;

  if(argc < 2) {
    fprintf(stderr,"USAGE reinforce inFile.csv (outFile.csv) (parameterFile)\n");
    exit(-9);
  }
  if(!(fp=fopen(argv[1],"r"))) {
    fprintf(stderr,"cannot read inputFile=[%s]\n",argv[1]);
    exit(-1);
  }

  fw = NULL;
  if(argc <= 3) {
    if(!(fw=fopen(argv[2],"w"))) {
      fprintf(stderr,"cannot read inputFile=[%s]\n",argv[1]);
      exit(-2);
    }
  }

  /* input data size */
  i=0;
  while(fgets(record,2048,fp)) {
    record[strlen(record)-1]='\0';
    if(record[0] == '$') continue;
    if(record[0] == '&') continue;
    if(record[0] == '#') continue;

    pc=strtok(record," ,\t");
    j=0;
    while(pc) {
      j++;
      pc = strtok(NULL," ,\n");
    }
    if(m < j) m=j;
    i++;
  }
  n=i;

  data=(double **)comMalloc(sizeof(double *)*n);
  for(i=0;i<n;i++) data[i]=(double *)comMalloc(sizeof(double)*m);

  /* data set */
  rewind(fp);
  i=0;
  while(fgets(record,2048,fp)) {
    record[strlen(record)-1]='\0';
    if(record[0] == '$') continue;
    if(record[0] == '&') continue;
    if(record[0] == '#') continue;

    pc=strtok(record," ,\t");
    j=0;
    while(pc) {
      data[i][j]=atof(pc);
      j++;
      pc = strtok(NULL," ,\t");
    }
    i++;
  }
  nSize=n;
  mSize=m;

  
  for(run=0; run<5; run++) {
    fprintf(stderr,"Beginning run #%d\n",run);
    for(i=0; i<N; i++) theta[i]= 0.0;                     // clear memory at start of each run
    for(episode_num=0; episode_num<100; episode_num++) {
      ret=episode(10000,NULL,data,nSize,mSize);
      //fprintf(stderr,"round=%d execute episode(10000)=%d \n",episode_num,ret);
    }
  }
  episode(10000,fw,data,nSize,mSize);

  /* free */
  for(i=0;i<n;i++) free(data[i]);
  free(data);

  return(0);
}

// Runs one episode of at most max_steps, returning episode length; see Figure 8.8 of RLAI book            
int episode(int max_steps,FILE *fw,double **data,int nsize,int msize) {

  int i,j,action,step,a;
  double reward,delta,temp;

  mcar_init(data,nsize,msize);                                               // initialize car's state
  for (i=0; i<N; i++) e[i] = 0.0;                            // clear all traces
  load_F();                                                  // compute features
  load_Qall();                                               // compute action values
  action = argmax(Q);                                        // pick argmax action
  if (with_probability(epsilon)) action = rand() % M;        // ...or maybe pick action at random
  step = 0;                                                  // now do a bunch of steps

  do {
    step++;
    for(i=0; i<N; i++) e[i] *= gamma*lambda;                 // let traces fall
    for(a=0; a<M; a++) {                                     // optionally clear other traces
      if(a != action) {
        for(j=0; j<NUM_TILINGS; j++) e[F[a][j]] = 0.0;
      }
    }
    for(j=0; j<NUM_TILINGS; j++) e[F[action][j]] =1.0;       // replace traces
    reward=mcar_step(action,step,fw,data,nsize,msize);                               // actually take action
    //reward = -1;
    delta = reward - Q[action];
    load_F();                                                // compute features new state
    load_Qall();                                             // compute new state values
    action = argmax(Q);
    if(with_probability(epsilon)) action = rand() % M;
    if(!mcar_goal_p(data,nsize,msize)) delta += gamma * Q[action];
    temp = (alpha/NUM_TILINGS)*delta;
    for(i=0; i<N; i++) theta[i] += temp * e[i];              // update theta (learn)
    load_Q(action);
    fprintf(stderr,"step=%d X=%d Y=%d Q[1]=%lf Q[2]=%lf Q[3]=%lf Q[4]=%lf\n",step,posX,posY,Q[0],Q[1],Q[2],Q[3]);
  }  while (!mcar_goal_p(data,nsize,msize) && step<max_steps);               // repeat until goal or time limit
  return(step);
}  
                                                             // return episode length
// Compute all the action values from current F and theta
void load_Qall() {
int a,j;

  for(a=0; a<M; a++) {
    Q[a] = 0;
    for (j=0; j<NUM_TILINGS; j++) Q[a] += theta[F[a][j]];
  }
}

// Compute an action value from current F and theta
void load_Q(int a) { 

   int j;
   Q[a] = 0;
   for(j=0; j<NUM_TILINGS; j++) Q[a] += theta[F[a][j]];
}

// Returns index (action) of largest entry in Q array, breaking ties randomly
int argmax(double Q[M]) {

   int best_action;
   double best_value;
   int a;
   double value;
   int num_ties;

   best_action = 0;
   best_value = Q[0];
   num_ties = 1;                    // actually the number of ties plus 1
   for( a=1; a<M; a++) {
     value = Q[a];
     if(value >= best_value) {
       if(value > best_value) {
         best_value = value;
         best_action = a;
       }
       else {
         num_ties++;
         if (0 == rand() % num_ties) {
           best_value = value;
           best_action = a;
         }
       }
     }
   }
   return(best_action);
}

// Returns TRUE with probability p    
bool with_probability(double p) {

   if(p > ((double)rand()) / RAND_MAX) return(1);
   else                                return(0);

   //return (p > ((double)rand()) / RAND_MAX);
}

    
///////////////  Mountain Car code begins here  ///////////////

// Mountain Car Global variables:
double mcar_position, mcar_velocity;

#define mcar_min_position -1.2
#define mcar_max_position 0.6
#define mcar_max_velocity 0.07            // the negative of this is also the minimum velocity
#define mcar_goal_position 0.5

#define POS_WIDTH (1.7 / 8)               // the tile width for position
#define VEL_WIDTH (0.14 / 8)              // the tile width for velocity

// Compute feature sets for current car state
void load_F() {

    double state_vars[2];
    int a;
#if 1
    state_vars[0]=posX;
	state_vars[1]=posY;
#else
    state_vars[0] = mcar_position / POS_WIDTH;
    state_vars[1] = mcar_velocity / VEL_WIDTH;
#endif
    for(a=0; a<M; a++) {
      GetTiles(&F[a][0],NUM_TILINGS,state_vars,2,N,a,-1,-1);
    }
}

// Initialize state of Car
void mcar_init(double **data,int nsize,int msize) {

#if 1
   posX=0;
   posY=0;
   goalX=msize-1;
   goalY=nsize-1;
#else
   mcar_position = -0.5;
   mcar_velocity = 0.0;
#endif
}

// Take action a, update state of car
double mcar_step(int a,int no,FILE *fw,double **data,int nsize,int msize) 
{

#if 1
   double dis;

   dis=0;
   switch(a) {
     case UP   :if(posY+1 >= nsize) dis=-1;
		        else
                if(data[posY+1][posX] == 1) dis=-2;
                else posY += 1;
                break;
     case DOWN :if(posY-1 <= 0) dis=-1;
		        else
                if(data[posY-1][posX] == 1) dis=-2;
				else posY -= 1;
                break;
     case LEFT :if(posX-1 <= 0) dis=-1;
		        else
                if(data[posY][posX-1] == 1) dis=-2;
				else posX -= 1;
                break;
     case RIGHT:if(posX+1 >= msize) dis=-1;
		        else
                if(data[posY][posX+1] == 1) dis=-2;
                else posX += 1;
                break;
   }


   if(fw) {
     fprintf(fw,"%d,%d,%d,%d\n",no,a,posX,posY);
   }

#if 1
   if(dis) return(dis);
   else    return(-1);
#else
   dis = goal_dist(posX,posY,goalX,goalY);

   if(dis > 0) return(1.0/dis);
   else        return(0);
#endif

#else
   mcar_velocity += (a-1)*0.001 + cos(3*mcar_position)*(-0.0025);
   if (mcar_velocity > mcar_max_velocity) mcar_velocity = mcar_max_velocity;
   if (mcar_velocity < -mcar_max_velocity) mcar_velocity = -mcar_max_velocity;
   mcar_position += mcar_velocity;
   if (mcar_position > mcar_max_position) mcar_position = mcar_max_position;
   if (mcar_position < mcar_min_position) mcar_position = mcar_min_position;
   if (mcar_position==mcar_min_position && mcar_velocity<0) mcar_velocity = 0;

   if(fw) {
     fprintf(fw,"%d,%d,%lf,%lf\n",no,a,mcar_velocity,mcar_position);
   }
#endif
}
double goal_dist()
{
    double dis;
	dis = pow(pow(posX - goalX,2) + pow(posY - goalY,2),0.5);
    return(dis);
}
// Is Car within goal region?
bool mcar_goal_p (double **data,int nsize,int msize) {

#if 1
   if(posX == goalX && posY == goalY) return(1);
   else                               return(0);
#else
   if(mcar_position >= mcar_goal_position) return(1);
   else                                    return(0);
#endif
}
