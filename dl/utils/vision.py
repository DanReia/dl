def convolution_output_calculator(
    input_dim: int, kernel_size: int, padding: int, stride: int
):
    return (((input_dim - kernel_size) + 2 * padding) / stride) + 1
