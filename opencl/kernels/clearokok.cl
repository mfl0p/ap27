/*

	clearokok kernel

*/


__kernel void clearokok(__global ulong *OKOK){


	int i = get_global_id(0);

	// clear array
	if(i < 23693){
		OKOK[i] = 0;
	}

}




