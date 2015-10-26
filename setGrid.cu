void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{
	// set your block dimensions and grid dimensions here
	gridDim.x = n / 64;
	gridDim.y = n / 16;

	if(n % blockDim.x != 0)
	{
		gridDim.x++;
	}

	if(n % blockDim.y != 0)
	{
		gridDim.y++;
	}
}
