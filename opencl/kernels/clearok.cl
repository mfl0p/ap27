/*

	clearok kernel

		also clear counters

*/





__kernel void clearok(__global char *OK, __global int *counter){


	int i = get_global_id(0);

	// clear array
	if(i < 23693){
		OK[i] = 1;
	}

	// clear counters
	if (i == 0){
		counter[1] = 0; // largest n count
		counter[2] = 0; // solutions
		counter[3] = 0; // PRP kernel overflow flag
	}

}




