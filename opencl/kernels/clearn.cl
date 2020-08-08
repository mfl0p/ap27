/*

	clearn kernel

*/


__kernel void clearn(__global int *counter){

	int i = get_global_id(0);

	if(i==0){
		counter[0]=0;
	}


}



