#define N_VARIABLE_NODES 16
#define N_CHECK_NODES 8
#define MAX_VARIABLE_EDGES 3
#define MAX_CHECK_EDGES 6

// This is for the local memory now
// This is a change
// Integer options
#define LLR_TYPE short
#define N_BITS_LLR (sizeof(LLR_TYPE)*8) // 8 bits per byte
#define MAX_LLR ((1 << (N_BITS_LLR-1))-1)
#define SIGN_BIT (MAX_LLR+1)
#define MAX_NEG_LLR (0-MAX_LLR)


typedef LLR_TYPE LLR;


// This structure holds all data required for the variable nodes. 
// Note that the last input will be the channel input for each variable node
// the n_inputs will always be n_checknodes+1
typedef struct VariableNodes
{
	int index;
	int checknode_indexes[MAX_VARIABLE_EDGES];


	LLR inputs[MAX_VARIABLE_EDGES+1];

	int n_inputs;
	int n_checknodes;
}VariableNode;

// This structure holds all data required for the check nodes
typedef struct CheckNodes
{
	int index;
	int variablenode_indexes[MAX_CHECK_EDGES];

	LLR inputs[MAX_CHECK_EDGES];


	int n_inputs;
	bool satisfied;
}CheckNode;

// There will just be one giant kernel that handles the entire execution path
__kernel __attribute__((reqd_work_group_size(N_CHECK_NODES,1,1))) void LDPCDecoder(__global VariableNode * restrict v_nodes, __global CheckNode * restrict c_nodes,int init,int writeback) {

	int workitem_id = get_global_id(0);
	
	// Now let's make local memory for the variable nodes
	__local VariableNode vnodes[N_VARIABLE_NODES];
	
	// Now let's make local memory for the check nodes
	__local CheckNode cnodes[N_CHECK_NODES];
	
	if(init) {
		// First we need to loadup our local memory with the variable node information
		vnodes[workitem_id] = v_nodes[workitem_id];
		vnodes[workitem_id + N_CHECK_NODES] = v_nodes[workitem_id + N_CHECK_NODES];
		
		cnodes[workitem_id] = c_nodes[workitem_id];
	}
	
	// Now that we only have N_CHECK_NODES threads, each thread needs to load two variable nodes
	
	
	CheckNode check = cnodes[workitem_id];
		

	// Let's now start the process
		// We need to find the first and second minimum
	LLR first_min = MAX_LLR;
	LLR second_min = MAX_LLR;
	LLR testinput;
	
	LLR test_first_min = ((((check.inputs[0]&MAX_LLR) < (check.inputs[1]&MAX_LLR) ? (check.inputs[0]&MAX_LLR) : (check.inputs[1]&MAX_LLR)) < ((check.inputs[2]&MAX_LLR) < (check.inputs[3]&MAX_LLR) ? (check.inputs[2]&MAX_LLR) : (check.inputs[3]&MAX_LLR)) ?  ((check.inputs[0]&MAX_LLR) < (check.inputs[1]&MAX_LLR) ? (check.inputs[0]&MAX_LLR) : (check.inputs[1]&MAX_LLR)) : ((check.inputs[2]&MAX_LLR) < (check.inputs[3]&MAX_LLR) ? (check.inputs[2]&MAX_LLR) : (check.inputs[3]&MAX_LLR))) < ((check.inputs[4]&MAX_LLR) < (check.inputs[5]&MAX_LLR) ? (check.inputs[4]&MAX_LLR) : (check.inputs[5]&MAX_LLR)) ? (((check.inputs[0]&MAX_LLR) < (check.inputs[1]&MAX_LLR) ? (check.inputs[0]&MAX_LLR) : (check.inputs[1]&MAX_LLR)) < ((check.inputs[2]&MAX_LLR) < (check.inputs[3]&MAX_LLR) ? (check.inputs[2]&MAX_LLR) : (check.inputs[3]&MAX_LLR)) ?  ((check.inputs[0]&MAX_LLR) < (check.inputs[1]&MAX_LLR) ? (check.inputs[0]&MAX_LLR) : (check.inputs[1]&MAX_LLR)) : ((check.inputs[2]&MAX_LLR) < (check.inputs[3]&MAX_LLR) ? (check.inputs[2]&MAX_LLR) : (check.inputs[3]&MAX_LLR))) : ((check.inputs[4]&MAX_LLR) < (check.inputs[5]&MAX_LLR) ? (check.inputs[4]&MAX_LLR) : (check.inputs[5]&MAX_LLR)));
	
	

	// Let's get the first minimum as well as the index of this minimum
	int min_index = 0;
	for(int input=0; input<check.n_inputs; input++)
	{
		// Remember to take the absolute value 
		// This is as simple as masking off the sign bit
		testinput = check.inputs[input] & MAX_LLR;

		if(testinput < first_min)
		{
			first_min = testinput;
			min_index = input;
		}
	}
	
	if(first_min != test_first_min) {
		printf("THEY'RE DIFFERENT!   %d    %d\n",test_first_min,first_min);
	}
	
	// Now we get the second minimum which means ignoring the minimum index
	for(int input =0; input < check.n_inputs; input++)
	{
		testinput = check.inputs[input] & MAX_LLR;

		if(input != min_index && testinput < second_min)
		{
			second_min = testinput;
		}
	}
	
	// Now we need to get the parity of the inputs
	// This will determine the sign of the outputs
	LLR parity = 0;
	parity =(check.inputs[0] & SIGN_BIT) ^ 
			(check.inputs[1] & SIGN_BIT) ^
			(check.inputs[2] & SIGN_BIT) ^ 
			(check.inputs[3] & SIGN_BIT) ^
			(check.inputs[4] & SIGN_BIT) ^
			(check.inputs[5] & SIGN_BIT);
			
	




	// Now that we have the parity, we can continue
	// This section will calculate what to place in each of the outputs 
	for(int input=0; input<check.n_inputs; input++)
	{
		// Let's get the variable node index we are dealing with
		int variableNode_index = check.variablenode_indexes[input];
		
		// Now we need to find out what input box number we are, in the variable node
		// we are looking at
		// This means finding the index based on the connections for that node
		int c_node_index = check.index;
		int input_index = 0;
		for(int i=0; i<vnodes[variableNode_index].n_checknodes; i++)
		{
			if(vnodes[variableNode_index].checknode_indexes[i] == c_node_index)
			{
				input_index = i;
			}
		}

		// Now we can move on to calculating the output
		LLR sign_b = parity;

		// The sign bit is assigned to complete the parity compared to the other inputs
		sign_b ^= check.inputs[input] & SIGN_BIT;

		LLR to_vnode;

		// The magnitude is decided as either the first or the second minimum depending on if we are
		// at the index of the first minimum.
		if(input != min_index)
			to_vnode = first_min;
		else
			to_vnode = second_min;


		// Before we send the data to the variable node, we need to convert it to 2's complement
		if(sign_b)
			to_vnode *= -1;

		// Send the information
		//v_nodes[variableNode_index].inputs[input_index] = to_vnode;
		
		// We write it to local memory unless the writeback bit has been set
		vnodes[variableNode_index].inputs[input_index] = to_vnode;
		if(writeback)
			v_nodes[variableNode_index].inputs[input_index] = to_vnode;

	}
	
		
	
	
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	
	// Now let's do the variable node stuff
	// Remember that each work-item now needs to do two variable nodes
	// We will do them simultaneously
	
	// Get the current variable node we are working on
	VariableNode variable = vnodes[workitem_id];
	VariableNode variable2 = vnodes[workitem_id + N_CHECK_NODES];
	
	// We should also load up the other information
	

	// Now we need to get the total sum
	LLR total = 0;
	LLR total2 = 0;
	int test;
	int test2;
	for(int input = 0; input<variable.n_inputs; input++)
	{
		// Add the current input to the total
		test = total + variable.inputs[input];
		test2 = total2 + variable2.inputs[input];
		
		// Check that the new total isn't going to cause an overflow with our LLR bits
		if(test > MAX_LLR) {
			total = MAX_LLR;
		}
		else if(test < MAX_NEG_LLR) {
			total = MAX_NEG_LLR;
		}
		else {
			total = test;
		}
		
		if(test2 > MAX_LLR) {
			total2 = MAX_LLR;
		}
		else if(test2 < MAX_NEG_LLR) {
			total2 = MAX_NEG_LLR;
		}
		else {
			total2 = test2;
		}
			
	}


	



	LLR current;
	LLR current2;
	// Now we need to subtract each of the checknodes individually
	for(int input = 0; input<variable.n_checknodes; input++)
	{
		// Get the index of the checknode of which we got a value and need to send one back
		int checkNode_index = variable.checknode_indexes[input];
		int checkNode_index2 = variable2.checknode_indexes[input];

		// We subtract off the current one to get the sum of all others
		current = total - variable.inputs[input];
		current2 = total2 - variable2.inputs[input];

		// Now we need to stuff that value back to the checknode
		// But first we need to find out which index of the checknode we belong to
		int v_index = variable.index;
		int input_index = 0;
		int v_index2 = variable2.index;
		int input_index2 = 0;
		
		// Iterate over all the checknode inputs to find ours
		for(int i=0; i<cnodes[checkNode_index].n_inputs; i++)
		{
			// If this check node index matches our index, this is the location to which we write in
			if(cnodes[checkNode_index].variablenode_indexes[i] == v_index)
			{
				input_index = i;
			}
			
			if(c_nodes[checkNode_index2].variablenode_indexes[i] == v_index2)
			{
				input_index2 = i;
			}
		}

		// Let's write the value into the checknode
		// But first we need to convert it back to sign and magnitude
		LLR mag=0;
		LLR sign_b = 0;
		LLR mag2 = 0;
		LLR sign_b2 = 0;
		if(current<0){
			current *= -1;
			sign_b = SIGN_BIT;
		}

		mag = current & MAX_LLR;
		
		if(current2<0) {
			current2 *= -1;
			sign_b2 = SIGN_BIT;
		}
		
		mag2 = current2 & MAX_LLR;

		// Write the value to the checknodes in sign and magnitude format

		cnodes[checkNode_index].inputs[input_index] = sign_b | mag;
		
		cnodes[checkNode_index2].inputs[input_index2] = sign_b2 | mag2;
		
		if(writeback) {
			c_nodes[checkNode_index].inputs[input_index] = sign_b | mag;
		
			c_nodes[checkNode_index2].inputs[input_index2] = sign_b2 | mag2;
		}
		

	}
}

