#define ALTERA
#include <iostream>
#include <fstream>
#include <ctime>
#include <math.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "E:\Andrew\Projects\common/ax_common.h"

#define N_VARIABLE_NODES 16
#define N_CHECK_NODES 8
#define MAX_VARIABLE_EDGES 3
#define MAX_CHECK_EDGES 6

// These options apply to the code being used as well as the number of iterations to do
int LDPC_SIZE = 16; // Note this is overwritten when a value is passed in via command line

#define FILENAME "matrix_"
#define OUTPUT_FILE "results_"
#define N_ITERATIONS 32


#define MAX_LINE 1024

static char * INPUT_FILENAME = (char*) malloc(sizeof(char)*MAX_LINE);
static char * OUTPUT_FILENAME = (char*) malloc(sizeof(char)*MAX_LINE);


// Integer options
#define LLR_TYPE cl_short
#define N_BITS_LLR (sizeof(LLR_TYPE)*8) // 8 bits per byte
#define MAX_LLR ((1 << (N_BITS_LLR-1))-1)
#define SIGN_BIT (MAX_LLR+1)
#define MAX_NEG_LLR (0-MAX_LLR)

#define BINS_PER_ONE_STEP 4
#define SATURATION_POINT int(MAX_LLR/BINS_PER_ONE_STEP)

typedef LLR_TYPE LLR;

//#define PRINT_DEBUG
#define QUICK_DEBUG

#ifdef PRINT_DEBUG
#undef QUICK_DEBUG
#endif

#define DEBUG_FILENAME "output_debug_integer.csv"
#define CHECKNODE_DEBUG_FILENAME "checknode_debug.csv"
#define VARIABLENODE_DEBUG_FILENAME "variablenode_debug.csv"




#pragma region OpenCL Specific Declarations
static char * KERNEL_FILENAME  = (char*) malloc(sizeof(char)*MAX_LINE);
static char const* COMPILER_OPTIONS = "-w";


#pragma endregion

#define DEBUG 1
#define AOCL_ALIGNMENT 64





#pragma region Error Checking Macros
#define SYSTEM_ERROR -1
#define CheckPointer(pointer) CheckPointerFunc(pointer,__FUNCTION__, __LINE__)
#define ExitOnErrorWMsg(msg) ExitOnErrorFunc(msg,__FUNCTION__,__LINE__)
#define ExitOnError() ExitOnErrorWMsg("Error! Please refer to")
#pragma endregion

#pragma region Structure Defintions
// This structure holds all data required for the variable nodes. 
// Note that the last input will be the channel input for each variable node
// the n_inputs will always be n_checknodes+1
typedef struct VariableNodes
{
	cl_int index;
	cl_int checknode_indexes[MAX_VARIABLE_EDGES];


	LLR inputs[MAX_VARIABLE_EDGES+1];

	cl_int n_inputs;
	cl_int n_checknodes;
}VariableNode;

// This structure holds all data required for the check nodes
typedef struct CheckNodes
{
	cl_int index;
	cl_int variablenode_indexes[MAX_CHECK_EDGES];

	LLR inputs[MAX_CHECK_EDGES];


	cl_int n_inputs;
	cl_bool satisfied;
}CheckNode;
#pragma endregion

#pragma region Function Declaractions
static void CheckPointerFunc(void * pointer,char const* const func,const int line);
static void ExitOnErrorFunc(char const* msg, char const* func, const int line);
static void readParityFile(VariableNode*&, CheckNode*&, int&, int&);
static bool checkIfSatisfied(CheckNode * c_nodes,const int n_checks,bool print);
static void updateParity(CheckNode *& c_nodes, const int n_checks);
static int * convertCodewordToBinary(VariableNode*&,const int n_vars);
static float* getCodeword(int size);
static LLR * convertToLLR(const float * codeword, const int n_vars);
static void printCodeword(const int * codeword, const int n_vars);
static void printResultsToFile(double overall, double variable, double check, double misc);
static void setFileNames();

double generate_random_gaussian();
double generate_random_number();
void introduceNoise(float*,int,int);

static void startOpenCL(cl_context,cl_device_id device, cl_program program);
static void startSimulationOpenCL(cl_context, cl_device_id,cl_program,VariableNode*&,CheckNode*&,const int, const int);
static void ldpcDecodeOpenCL(cl_context, cl_command_queue,cl_kernel,VariableNode*&,CheckNode*&,const int, const int,float*);
void initializeNodes(VariableNode*&, CheckNode*&,const LLR*,const int, const int);
#pragma endregion






int main(int argc, char * argv[])
{
	if(argc != 2){
		printf("No LDPC size supplied to the program, using %d\n",LDPC_SIZE);
	}
	else {
		// Otherwise we were supplied a codelength for the LDPC, let's read it
		LDPC_SIZE = atoi(argv[1]);
		printf("LDPC input size of %d being used\n",LDPC_SIZE);
	}

	// Let's do a quick check on the number
	if(LDPC_SIZE <=0) {
		printf("Codelength (number of variable nodes) is not valid\n");
		ExitOnError();
	}

	setFileNames();


	// Set up the program
	cl_context context;
	cl_device_id device;
	cl_program program;
	cl_device_type type = CL_DEVICE_TYPE_ALL;

	printf("\nOpenCL API\n\n");
	printf("Max LLR = %d\n",MAX_LLR);
	AxCreateContext(type,AX_PLATFORM_STRING,&context,&device);
#ifdef ALTERA
	AxBuildProgramAltera(&program,device,context,KERNEL_FILENAME,COMPILER_OPTIONS);
#else
	AxBuildProgram(&program,device,context,KERNEL_FILENAME,COMPILER_OPTIONS);
#endif

	startOpenCL(context,device,program);

	clReleaseProgram(program);
	clReleaseContext(context);


}

void startOpenCL(cl_context context, cl_device_id device, cl_program program)
{
	// Declare the variables
	// Declare an array to hold the checknodes
	CheckNode * c_nodes=NULL;

	// Declare an array to hold the variablenodes
	VariableNode * v_nodes=NULL;

	// The number of check nodes and variable nodes
	int n_checks=0;
	int n_vars=0;

	// Let's read the parity file to get the nodes
	readParityFile(v_nodes,c_nodes,n_vars,n_checks);

	// Let's just do a few quick checks to make sure that the read worked properly
	// We should check the c_nodes and v_nodes pointers
	// We also need to check that the number of variable nodes and check nodes are valid
	CheckPointer(c_nodes);CheckPointer(v_nodes);
	if(n_vars <= 0 || n_checks <= 0)
		ExitOnErrorWMsg("Number of variables and check nodes not valid.");

	printf("\n*********************************************\n");
	printf("Variable Nodes %d, Check Nodes %d\n",n_vars,n_checks);
	printf("%d Log Likelihood Ratio Bits\n",N_BITS_LLR);
	printf("Max Variable Edges %d, Max Check Edges %d\n",MAX_VARIABLE_EDGES,MAX_CHECK_EDGES);

	// Now we are ready to start our simulation
	startSimulationOpenCL(context,device,program,v_nodes,c_nodes,n_vars,n_checks);

}

void startSimulationOpenCL(cl_context context, cl_device_id device,cl_program program,VariableNode*& v_nodes,CheckNode*& c_nodes, const int n_vars, const int n_checks)
{

	// Firstly, we need to get our codeword to decode
	float * codeword = getCodeword(n_vars);
	cl_int err;
	// Let's create our openCL stuff

	// Now we need to create our command queues
	// We will create one command queue for each kernel
	cl_command_queue LDPCqueue = clCreateCommandQueue(context,device,0,&err);AxCheckError(err);


	// Now we need to create the kernels themselves.
	cl_kernel LDPCkernel = clCreateKernel(program,"LDPCDecoder",&err);AxCheckError(err);




	// Now let's simply start the LDPC
	ldpcDecodeOpenCL(context, LDPCqueue, LDPCkernel,v_nodes,c_nodes,n_vars,n_checks,codeword);
}

void ldpcDecodeOpenCL(cl_context			context,
					  cl_command_queue		LDPCqueue,
					  cl_kernel				LDPCkernel,
					  VariableNode*&		v_nodes,
					  CheckNode*&			c_nodes,
					  const int				n_vars,
					  const int				n_checks,
					  float*				codeword)
{
	cl_int err;


	// Note this is where I would initiate an LLR for fixed integer

	// Now we need to introduce some noise into our codeword
	// Note that this function changes our codeword input variable
	introduceNoise(codeword,20,n_vars);

	// Now I need to add in the step to conver the number to an integer
	LLR * codeword_LLR = convertToLLR(codeword,n_vars);


	// Now let's load up the inputs for the nodes for the first iteration
	initializeNodes(v_nodes,c_nodes,codeword_LLR,n_vars,n_checks);

#ifdef PRINT_DEBUG
	FILE* debug = fopen(DEBUG_FILENAME,"w");CheckPointer(debug);
	FILE* checknodedebug = fopen(CHECKNODE_DEBUG_FILENAME,"w");CheckPointer(checknodedebug);
	FILE* variablenodedebug = fopen(VARIABLENODE_DEBUG_FILENAME,"w");CheckPointer(variablenodedebug);
	printf("Press any key to write the initial values\n");
	getchar();

	fprintf(debug,"Variable Nodes:\n");
	fprintf(debug,"Node Type,Node Number,Input Number,Check Node,Value,Floating Value\n");

	for(int i=0; i<n_vars; i++)
	{
		for(int j=0; j<v_nodes[i].n_inputs; j++) {
			fprintf(debug, "%s,%d,%d,%d,%d,%f\n","Variable",i,j,v_nodes[i].checknode_indexes[j],v_nodes[i].inputs[j],codeword[i]);
		}
	}

	fprintf(debug,"\nCheck Nodes:\n");
	fprintf(debug,"%s,%s,%s,%s,%s,%s\n","Node Type","Node Number","Input Number","Variable Node","Sign","Magnitude");
	for(int i=0; i<n_checks; i++)
	{
		for(int j=0; j<c_nodes[i].n_inputs; j++) {
			fprintf(debug, "%s,%d,%d,%d,%d,%d\n","Check",i,j,c_nodes[i].variablenode_indexes[j],(c_nodes[i].inputs[j]&SIGN_BIT)>>15,(c_nodes[i].inputs[j]&MAX_LLR));
		}
	}
	fflush(debug);
	fclose(debug);

#endif
#ifdef QUICK_DEBUG
	FILE* debug = fopen(DEBUG_FILENAME,"w");
	printf("Quick Debug started with in file '%s'\n",DEBUG_FILENAME);
	fprintf(debug,"This file is a result from the Quick Debug setting.  The values for each iteration represent the decimal version of the check node satisifed booleans.\n");
	fprintf(debug,"In other words, the check nodes are checked for their satisfying parity conditions and the binary results from all the checknodes are combined into a single decimal number.\n");
	fprintf(debug,"In order to verify the accuracy of this LDPC you will need to compare the below values with that of the Floating Point version or another previously verified LDPC\n\n");
#endif


	// Now let's start the kernel stuff
	long start_overall = clock();
	long start_variable,end_variable,start_check,end_check;
	long variable_total=0;
	long check_total=0;

	size_t N_BYTES_CHECK = sizeof(CheckNode)*n_checks;
	size_t N_BYTES_VARIABLE = sizeof(VariableNode)*n_vars;

	// Let's load up the buffers
	cl_mem check_nodes = clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,N_BYTES_CHECK,c_nodes,&err);AxCheckError(err);
	cl_mem variable_nodes = clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,N_BYTES_VARIABLE,v_nodes,&err);AxCheckError(err);

	


	// Now let's set the arguments for the checknode kernel
	int first = 1;
	int writeback = 0;
#if defined(QUICK_DEBUG) | defined(PRINT_DEBUG)
	writeback = 1;
#endif
	AxCheckError(clSetKernelArg(LDPCkernel,0,sizeof(variable_nodes),&variable_nodes));
	AxCheckError(clSetKernelArg(LDPCkernel,1,sizeof(check_nodes),&check_nodes));



	size_t offset_variable = 0;
	size_t global_work_size_variablenode = n_vars;
	size_t local_work_size_variablenode = n_vars;

	printf("Starting kernel iterations\n");
	for(int iter=0; iter<N_ITERATIONS; iter++)
	{
#ifdef PRINT_DEBUG
		printf("Press any key to start Iteration %d check nodes\n",iter);
		getchar();
		debug = fopen(DEBUG_FILENAME,"a");
		checknodedebug = fopen(CHECKNODE_DEBUG_FILENAME,"a");CheckPointer(checknodedebug);
		variablenodedebug = fopen(VARIABLENODE_DEBUG_FILENAME,"a");CheckPointer(variablenodedebug);
		fprintf(debug,"\n\nIteration %d:\n",iter);
		fprintf(checknodedebug,"\n\nIteration %d:\n",iter);
		fprintf(variablenodedebug,"\n\nIteration %d:\n",iter);
#endif
		start_check = clock();
		// Start the CheckNode Kernel
		AxCheckError(clEnqueueNDRangeKernel(LDPCqueue,LDPCkernel,1,NULL,&global_work_size_variablenode,&local_work_size_variablenode,0,NULL,NULL));
		//clFinish(checkNodeQueue);
		//printf("Check node started\n");
		clFinish(LDPCqueue);
		end_check = clock();


#ifdef PRINT_DEBUG

		// Now let's read back the data
		AxCheckError(clEnqueueReadBuffer(checkNodeQueue,check_nodes,CL_TRUE,0,N_BYTES_CHECK,c_nodes,0,NULL,NULL));
		AxCheckError(clEnqueueReadBuffer(variableNodeQueue,variable_nodes,CL_TRUE,0,N_BYTES_VARIABLE,v_nodes,0,NULL,NULL));

		// Now that we have the data we should debug it
		// We want to print out everything check node related
		for(int shownode = 0; shownode<n_checks; shownode++) {
			fprintf(checknodedebug,"Showing info for node %d:\n",shownode);
			fprintf(checknodedebug,"%s,%s,%s,%s,%s,%s,%s\n","Node Type","Node Number","Input Number","Variable Node","Sign Input","Mag Input","Value Output");
			for(int i=0; i<c_nodes[shownode].n_inputs; i++) {
				int check_index=0;
				for(int j=0; j<v_nodes[c_nodes[shownode].variablenode_indexes[i]].n_checknodes; j++) {
					if(v_nodes[c_nodes[shownode].variablenode_indexes[i]].checknode_indexes[j] == shownode) {
						check_index = j;//v_nodes[c_nodes[shownode].variablenode_indexes[i]].checknode_indexes[j];
					}
				}
				fprintf(checknodedebug,"%s,%d,%d,%d,%d,%d,%d\n","Check",shownode,i,c_nodes[shownode].variablenode_indexes[i],(c_nodes[shownode].inputs[i]&SIGN_BIT)>>(N_BITS_LLR-1),c_nodes[shownode].inputs[i]&MAX_LLR,v_nodes[c_nodes[shownode].variablenode_indexes[i]].inputs[check_index]);
			}
		}
		fprintf(debug,"Checking if nodes are satisified (sat = satisfied)\n");
		fprintf(debug,",%s,%s,%s,%s,%s,%s,%s,%s\n,","Node 0","Node 1","Node 2","Node 3", "Node 4","Node 5","Node 6","Node 7");
		// Now let's see which checknodes are satisified
		for(int node= 0; node<n_checks; node++) {
			CheckNode * check = &c_nodes[node];

			int parity = 0;
			for(int input=0; input<(*check).n_inputs; input++) {
				if((*check).inputs[input] < 0)
					parity ^= 1;
			}

			if(parity)
				fprintf(debug,"Not,");
			else
				fprintf(debug,"Sat,");

		}
#endif

#ifdef PRINT_DEBUG

		AxCheckError(clEnqueueReadBuffer(variableNodeQueue,variable_nodes,CL_TRUE,0,N_BYTES_VARIABLE,v_nodes,0,NULL,NULL));
		AxCheckError(clEnqueueReadBuffer(checkNodeQueue,check_nodes,CL_TRUE,0,N_BYTES_CHECK,c_nodes,0,NULL,NULL));
		
		// We want to print out everything check node related
		for(int shownode = 0; shownode<n_vars; shownode++) {
			fprintf(variablenodedebug,"\n\nShowing info for node %d:\n",shownode);
			fprintf(variablenodedebug,"%s,%s,%s,%s,%s,%s,%s\n","Node Type","Node Number","Input Number","Check Node","Value Input","Sign Output","Mag Output");
			int i;
			for(i=0; i<v_nodes[shownode].n_checknodes; i++) {
				int variable_index=0;
				for(int j=0; j<c_nodes[v_nodes[shownode].checknode_indexes[i]].n_inputs; j++) {
					if(c_nodes[v_nodes[shownode].checknode_indexes[i]].variablenode_indexes[j] == shownode) {
						variable_index = j;//c_nodes[v_nodes[0].checknode_indexes[i]].variablenode_indexes[j];
						break;
					}
				}
				fprintf(variablenodedebug,"%s,%d,%d,%d,%d,%d,%d\n","Variable",shownode,i,v_nodes[shownode].checknode_indexes[i],(v_nodes[shownode].inputs[i]),(c_nodes[v_nodes[shownode].checknode_indexes[i]].inputs[variable_index]&SIGN_BIT)>>(N_BITS_LLR-1),c_nodes[v_nodes[shownode].checknode_indexes[i]].inputs[variable_index]&MAX_LLR);

			}
			fprintf(variablenodedebug,"%s,%d,%s,%s,%d,%s,%s\n","Variable",shownode,"Channel","N/A",(v_nodes[shownode].inputs[i]),"N/A","N/A");
		}
		fflush(debug);fflush(variablenodedebug);fflush(checknodedebug);
		fclose(debug);
		fclose(variablenodedebug);
		fclose(checknodedebug);
#endif
#ifdef QUICK_DEBUG
		AxCheckError(clEnqueueReadBuffer(LDPCqueue,check_nodes,CL_TRUE,0,N_BYTES_CHECK,c_nodes,0,NULL,NULL));
		AxCheckError(clEnqueueReadBuffer(LDPCqueue,variable_nodes,CL_TRUE,0,N_BYTES_VARIABLE,v_nodes,0,NULL,NULL));

		// Now that we have the buffers we can calculate the satisification conditions

		int quick_output = 0;
		for(int node= 0; node<n_checks; node++) {
			CheckNode * check = &c_nodes[node];

			int parity = 0;
			for(int input=0; input<(*check).n_inputs; input++) {
				parity ^= (*check).inputs[input]&SIGN_BIT;
			}
			parity >>= (N_BITS_LLR -1);

			quick_output <<= 1;
			quick_output |= parity;
		}
		fprintf(debug,"%d,%d\n",iter,quick_output);
#endif

		check_total += end_check - start_check;
	}
#ifdef QUICK_DEBUG
	fflush(debug);
	fclose(debug);
#endif



	// Now let's read back the data
	AxCheckError(clEnqueueReadBuffer(LDPCqueue,check_nodes,CL_TRUE,0,N_BYTES_CHECK,c_nodes,0,NULL,NULL));
	AxCheckError(clEnqueueReadBuffer(LDPCqueue,variable_nodes,CL_TRUE,0,N_BYTES_VARIABLE,v_nodes,0,NULL,NULL));

	long end_overall = clock();

	double overall = ((double)end_overall - start_overall)/CLOCKS_PER_SEC;
	double variable = ((double)0)/CLOCKS_PER_SEC;
	double check = ((double)check_total)/CLOCKS_PER_SEC;
	double misc = ((((double)end_overall - start_overall)/CLOCKS_PER_SEC) - (((double)variable_total)/CLOCKS_PER_SEC + (double)check_total/CLOCKS_PER_SEC));

	printf("Overall Time = %f sec\n",((double)end_overall - start_overall)/CLOCKS_PER_SEC);
	printf("Variable Time = %f sec\n",((double)variable_total)/CLOCKS_PER_SEC);
	printf("Check Time = %f sec\n",((double)check_total)/CLOCKS_PER_SEC);
	printf("Misc Time = %f sec\n",((((double)end_overall - start_overall)/CLOCKS_PER_SEC) - (((double)variable_total)/CLOCKS_PER_SEC + (double)check_total/CLOCKS_PER_SEC)));

	checkIfSatisfied(c_nodes,n_checks,true);

	printResultsToFile(overall,variable,check,misc);


	//printCodeword(convertCodewordToBinary(v_nodes,n_vars),n_vars);



}

void setFileNames() {

	strncpy(INPUT_FILENAME,FILENAME,MAX_LINE);
	_snprintf(INPUT_FILENAME,MAX_LINE,"%s%d.txt",INPUT_FILENAME,LDPC_SIZE);

	strncpy(OUTPUT_FILENAME,OUTPUT_FILE,MAX_LINE);
	_snprintf(OUTPUT_FILENAME,MAX_LINE,"%s%d.txt",OUTPUT_FILENAME,LDPC_SIZE);

#ifdef ALTERA
	_snprintf(KERNEL_FILENAME,MAX_LINE,"kernels_%d.aocx",LDPC_SIZE);
#else
	_snprintf(KERNEL_FILENAME,MAX_LINE,"../kernels_%d.cl",LDPC_SIZE);
#endif
}

void printResultsToFile(double overall, double variable, double check, double misc) {
	// Now let's open the file
	FILE * results = fopen(OUTPUT_FILENAME,"w"); CheckPointer(results);

	// Now let's write data to the file
	fprintf(results,"%f %f %f %f\n",overall,variable,check,misc);

	// Now we close the file
	fclose(results);

}





// Convert the input BPSK codeword into a binary integer codeword
int * convertCodewordToBinary(VariableNode*& v_nodes,const int n_vars)
{
	int * codeword = (int*) malloc(sizeof(int)*n_vars);
	LLR total;
	for(int node=0; node<n_vars; node++)
	{
		total = 0;
		int test = 0;
		for(int input=0; input<v_nodes[node].n_inputs; input++) {
			test = total + v_nodes[node].inputs[input];
			if(test > MAX_LLR) total = MAX_LLR;
			else if(test < MAX_NEG_LLR) total = MAX_NEG_LLR;
			else total = test;

		}
		if(total<0)
			codeword[node] =1;
		else
			codeword[node] =0;
	}

	return codeword;
}

// Check if all of the check nodes are satisfied
bool checkIfSatisfied(CheckNode * c_nodes,const int n_checks,bool print)
{
	bool output = true;
	// Iterate over all nodes
	for(int node=0; node<n_checks; node++)
	{
		CheckNode * check = &c_nodes[node];

		LLR parity = 0;
		// Now update the parity for this node
		for(int input=0; input<(*check).n_inputs; input++)
		{
			parity ^= (*check).inputs[input] & SIGN_BIT;
		}

		// Now update the boolean
		if(print)
		{
			if(parity)
				printf("Check Node %d not satisfied\n",node);
		}

		if(parity)
			output = false;
	}

	if(output && print)
		printf("All check nodes satisfied!\n");

	return output;

}

#pragma region Codeword Operations
// Get the codeword in BPSK 
float* getCodeword(int size)
{
	float *codeword = (float*)malloc(sizeof(float)*size);
	int * input = (int*) malloc(sizeof(int)*size);
	for(int i=0; i<size; i++)
		input[i] = 0;

	input[0] = 1;

	// Convert it to BPSK
	// 0 -> 1, 1 -> -1
	// Map binary 0 to 1
	// May binary 1 to -1
	for(int i=0; i<size; i++)
	{
		if(input[i])
			codeword[i] = -1;
		else
			codeword[i] = 1;
	}

	return codeword;
}

// Introduce gaussian noise to the input codeword based on the given SNR
void introduceNoise(float* codeword, int SNR, int n_vars)
{
	double sigma = pow(10,(double)-SNR/20);

	for(int i=0; i<n_vars; i++)
	{
		codeword[i] += generate_random_gaussian()*sigma;
		//printf("Codeword[%d] = %f\n",i,codeword[i]);
	}
}

void printCodeword(const int * codeword, const int n_vars) {
	for(int i=0; i<n_vars; i++) {
		printf("codeword[%d] = %d\n",i,codeword[i]);
	}

}

/* This function will take in the given float version of the codeword
and convert it to an integer representation LLR.  Please note that
it converts the LLR into a 2's complement number first
*/
LLR * convertToLLR(const float * codeword, const int n_vars) {
	LLR * output = (LLR*) malloc(sizeof(LLR)*n_vars);

	// Iterate over the entire codeword
	for(int i=0; i<n_vars; i++) {
		float word = abs(codeword[i]);
		output[i] = 0;

		// Now let's iterate over the possible bins and see where it fits
		float division_point = ((float)MAX_LLR)/SATURATION_POINT ;
		//printf("division_point = %f\n",division_point);
		for(int j=1; j<=(MAX_LLR); j++) {

			if(word <  j / division_point) {
				output[i] = j;
				break;
			}

		}
		//printf("codeword[%d] = %f	output[%d] = %u    sign = %d\n",i,codeword[i],i,output[i],(output[i]&SIGN_BIT)>>15);

		// Once we have the positioning we just need to make sure it's still within the valid LLR
		if(word >=  MAX_LLR / division_point) {
			output[i] = MAX_LLR;
		}

		// Now set the sign bit
		if(codeword[i] < 0) {
			output[i] *= -1;
		}

	}
	return output;
}
#pragma endregion


// This function will read the parity check matrix A-list file and get the following information:
// - number of variable nodes, number of check nodes
// - an array of variablenode structures
// - an array of checknode structures
// PLEASE NOTE ALL VARIABLES PASSED INTO THIS FUNCTION ARE PASSED BY REFERENCE AND WILL BE MODIFIED
// Please pass this function empty variables to store the result
void readParityFile(VariableNode *& v_nodes, CheckNode *& c_nodes,int &n_vars,int &n_checks)
{
#define MAX_LINE 1024
	char line[MAX_LINE];

	// Open the file
	FILE * parity_matrix = fopen(INPUT_FILENAME,"r");
	CheckPointer(parity_matrix);

	// Now let's start reading
	// The first two values in the text file will be the number of variable nodes and the number of check nodes
	if(fscanf(parity_matrix,"%d %d",&n_vars,&n_checks)!=2)
		ExitOnErrorWMsg("Error with parity_matrix A-list file format,");

	// Now that we know the number of variable nodes and checks we can allocate the amounts
	v_nodes = (VariableNode*)_aligned_malloc(sizeof(VariableNode)*n_vars,AOCL_ALIGNMENT);
	c_nodes = (CheckNode*)_aligned_malloc(sizeof(CheckNode)*n_checks,AOCL_ALIGNMENT);

	// Throw away the next line.  This line tells us the approximate number of connections for each node, but we want the more accurate ones
	int temp;
	if(fscanf(parity_matrix,"%d %d",&temp,&temp)!=2)
		ExitOnErrorWMsg("Missing second line in A-list file.");

	// Now let's get to the good stuff
	// According to the A-list format the next line of text will be the list of check nodes for each variable node in order
	// The next line will be a list of check nodes for each variable node
	// Therefore we need to iterate n_vars times to get all of the connections
	for(int i=0; i<n_vars; i++)
	{
		int num_per_variable;
		if(fscanf(parity_matrix,"%d",&num_per_variable)!=1)
			ExitOnErrorWMsg("Error with parity_matrix A-list file format.");

		// Store the number of check nodes for each variable node and the number of inputs
		v_nodes[i].n_checknodes = num_per_variable;
		v_nodes[i].n_inputs = num_per_variable+1; // Add one here as there is a spot for the initial channel measurement

		// Let's also store the index
		v_nodes[i].index = i;
	}

	// The next line will be a list of variable nodes for each check node
	// Therefore we need to iterate n_checks times to get all of the connections
	for(int i=0; i<n_checks; i++)
	{
		int num_per_check;
		if(fscanf(parity_matrix,"%d",&num_per_check)!=1)
			ExitOnErrorWMsg("Error with parity_matrix A-list file format.");

		// Store the number of inputs to the check node
		c_nodes[i].n_inputs = num_per_check; // Note there is not an extra here as there is no channel measurement

		// Also store the index
		c_nodes[i].index = i;

		// Set the satisifed bit to false
		c_nodes[i].satisfied = false;
	}

	// Next we get the indexes for variable nodes (edges)
	// This is the most important section where we determine which nodes get connected to each other
	// First up are the variable node connections.  So we need to iterate over all the variable nodes
	for(int i=0; i<n_vars; i++)
	{
		int position;

		// Now we can use the number of checknodes we just received from the previous lines
		// We need to store which checknodes this variable node is connected to
		for(int j=0; j<v_nodes[i].n_checknodes; j++)
		{
			if(fscanf(parity_matrix,"%d",&position)!=1)
				ExitOnErrorWMsg("Error with parity_matrix A-list file format.");

			// Now store the position
			v_nodes[i].checknode_indexes[j] = position-1; // Note we need to subtract one here to start the position at 0
		}
	}


	// Same thing, but now for the check nodes. 
	// Next we get the indexes for the check nodes (edges)
	for(int i=0; i<n_checks; i++)
	{
		int position;
		for(int j=0; j<c_nodes[i].n_inputs; j++)
		{
			if(fscanf(parity_matrix,"%d",&position)!=1)
				ExitOnErrorWMsg("Error with parity_matrix A-list file format.");

			// Now store the position
			c_nodes[i].variablenode_indexes[j] = position-1; // Note we need to subtract one here to start the position at 0
		}
	}

	// That concludes the reading of the parity matrix file
	// All data is now stored within the v_nodes and c_nodes arrays
	fclose(parity_matrix);
	printf("Parity matrix file %s read successfully\n",INPUT_FILENAME);
	return;
}


//Initialize the nodes for the first run
void initializeNodes(VariableNode *& v_nodes, CheckNode *& c_nodes, const LLR * codeword, const int n_vars, const int n_checks) 
{

	// First we load up the channel measurement into all of the variable nodes
	// Iterate over each variable node
	for(int i=0; i<n_vars; i++)
	{
		// Now for each variable node, let's load up all of the connections with the channel connections
		for(int j=0; j<v_nodes[i].n_inputs; j++)
		{
			v_nodes[i].inputs[j] = codeword[i];
		}
	}


	// Load up the inputs for the check nodes for the first iteration
	// This means we are now passing the specific channel measurements for each connection
	for(int i=0; i<n_checks; i++)
	{
		CheckNode *currentCheckNode = &c_nodes[i];

		// For each one of the inputs in this check node, we need to load up the information from the variable node
		for(int j=0; j<c_nodes[i].n_inputs; j++)
		{
			int variable_index = c_nodes[i].variablenode_indexes[j];
			int index = v_nodes[variable_index].n_inputs-1; // All of the spots hold the same value currently

			// We have to convert this value into sign and mag
			LLR sign_and_mag_ver = v_nodes[variable_index].inputs[index];
			if(sign_and_mag_ver < 0) {
				sign_and_mag_ver *= -1;
				sign_and_mag_ver |= SIGN_BIT;
			}

			// Let's store the value in our input
			(*currentCheckNode).inputs[j] = sign_and_mag_ver;
		}
	}
}

#pragma region Error Checking Functions
// This function will check the given input pointer for validaty and fail exit the program if necessary
void CheckPointerFunc(void * pointer,char const* const func, const int line)
{
	if(pointer == NULL)
	{
		fprintf(stderr,"Error with pointer in %s line %d\n",func,line);
		exit(SYSTEM_ERROR);
	}
}

void ExitOnErrorFunc(char const* msg, char const* func, const int line)
{
	fprintf(stderr,"%s %s line %d\n",msg,func,line);
	exit(SYSTEM_ERROR);
}
#pragma endregion

#pragma region Random Number Code
double generate_random_gaussian()
{
#define PI 3.14159265358979323846
	double U1 = generate_random_number();//(double)rand()/((doule)RAND_MAX+1);
	double U2 = generate_random_number();//(double)rand()/((double)RAND_MAX+1);

	double Z0 = sqrt(-2*log(U1))*cos(2*PI*U2);
	return Z0;
}

double generate_random_number()
{
#define C1 4294967294
#define C2 4294967288
#define C3 4294967280
#define MAX 4294967295


	static unsigned int s1 = rand();
	static unsigned int s2 = rand();
	static unsigned int s3 = rand();

	// Here's the first part
	// Now let's crank the tausworthe
	unsigned int xor1 = s1 << 13;

	// The first set of and/xor
	unsigned int left_temp = xor1 ^ s1;
	unsigned int right_temp = C1 & s1;

	// Shifts
	left_temp = left_temp >> 19;
	right_temp = right_temp << 12;

	s1 = left_temp ^ right_temp;

	// Second part
	xor1 = s2<<2;
	left_temp = xor1 ^ s2;
	right_temp = C2 & s2;

	left_temp = left_temp >> 25;
	right_temp = right_temp << 4;

	s2 = left_temp ^ right_temp;

	// Third part
	xor1 = s3 << 3;
	left_temp = xor1 ^ s3;
	right_temp = C3 & s3;

	left_temp = xor1 ^ s3;
	right_temp = C3 & s3;

	left_temp = left_temp >> 11;
	right_temp = right_temp << 17;

	s3 = left_temp ^ right_temp;

	// Now the return
	unsigned int output = s1 ^ s2 ^ s3;

	// Now just convert it into a double
	double last_value = (double)output /MAX;

	//cout<<"last value" << last_value<<endl;
	return last_value;

}
#pragma endregion


