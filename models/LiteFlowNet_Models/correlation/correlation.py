import torch

import cupy
import math
import re

class Stream:
	ptr = torch.cuda.current_stream().cuda_stream
# end

kernel_Correlation_rearrange = '''
	extern "C" __global__ void kernel_Correlation_rearrange(
		const int n,
		const float* input,
		float* output
	) {
	  int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	  if (intIndex >= n) {
	    return;
	  }

	  int intSample = blockIdx.z;
	  int intChannel = blockIdx.y;

	  float dblValue = input[(((intSample * SIZE_1(input)) + intChannel) * SIZE_2(input) * SIZE_3(input)) + intIndex];

	  __syncthreads();

	  int intPaddedY = (intIndex / SIZE_3(input)) + 3*{{intStride}};
	  int intPaddedX = (intIndex % SIZE_3(input)) + 3*{{intStride}};
	  int intRearrange = ((SIZE_3(input) + 6*{{intStride}}) * intPaddedY) + intPaddedX;

	  output[(((intSample * SIZE_1(output) * SIZE_2(output)) + intRearrange) * SIZE_1(input)) + intChannel] = dblValue;
	}
'''

kernel_Correlation_updateOutput = '''
	extern "C" __global__ void kernel_Correlation_updateOutput(
	  const int n,
	  const float* rbot0,
	  const float* rbot1,
	  float* top
	) {
	  extern __shared__ char patch_data_char[];
	  
	  float *patch_data = (float *)patch_data_char;
	  
	  // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
	  int x1 = (blockIdx.x + 3) * {{intStride}};
	  int y1 = (blockIdx.y + 3) * {{intStride}};
	  int item = blockIdx.z;
	  int ch_off = threadIdx.x;
	  
	  // Load 3D patch into shared shared memory
	  for (int j = 0; j < 1; j++) { // HEIGHT
	    for (int i = 0; i < 1; i++) { // WIDTH
	      int ji_off = (j + i) * SIZE_3(rbot0);
	      for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	        int idx1 = ((item * SIZE_1(rbot0) + y1+j) * SIZE_2(rbot0) + x1+i) * SIZE_3(rbot0) + ch;
	        int idxPatchData = ji_off + ch;
	        patch_data[idxPatchData] = rbot0[idx1];
	      }
	    }
	  }
	  
	  __syncthreads();
	  
	  __shared__ float sum[32];
	  
	  // Compute correlation
	  for (int top_channel = 0; top_channel < SIZE_1(top); top_channel++) {
	    sum[ch_off] = 0;
	  
	    int s2o = (top_channel % 7 - 3) * {{intStride}};
	    int s2p = (top_channel / 7 - 3) * {{intStride}};
	    
	    for (int j = 0; j < 1; j++) { // HEIGHT
	      for (int i = 0; i < 1; i++) { // WIDTH
	        int ji_off = (j + i) * SIZE_3(rbot0);
	        for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	          int x2 = x1 + s2o;
	          int y2 = y1 + s2p;
	          
	          int idxPatchData = ji_off + ch;
	          int idx2 = ((item * SIZE_1(rbot0) + y2+j) * SIZE_2(rbot0) + x2+i) * SIZE_3(rbot0) + ch;
	          
	          sum[ch_off] += patch_data[idxPatchData] * rbot1[idx2];
	        }
	      }
	    }
	    
	    __syncthreads();
	    
	    if (ch_off == 0) {
	      float total_sum = 0;
	      for (int idx = 0; idx < 32; idx++) {
	        total_sum += sum[idx];
	      }
	      const int sumelems = SIZE_3(rbot0);
	      const int index = ((top_channel*SIZE_2(top) + blockIdx.y)*SIZE_3(top))+blockIdx.x;
	      top[index + item*SIZE_1(top)*SIZE_2(top)*SIZE_3(top)] = total_sum / (float)sumelems;
	    }
	  }
	}
'''

kernel_Correlation_updateGradFirst = '''
	#define ROUND_OFF 50000

	extern "C" __global__ void kernel_Correlation_updateGradFirst(
	  const int n,
	  const int intSample,
	  const float* rbot0,
	  const float* rbot1,
	  const float* gradOutput,
	  float* gradFirst,
	  float* gradSecond
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	  int n = intIndex % SIZE_1(gradFirst); // channels
	  int l = (intIndex / SIZE_1(gradFirst)) % SIZE_3(gradFirst) + 3*{{intStride}}; // w-pos
	  int m = (intIndex / SIZE_1(gradFirst) / SIZE_3(gradFirst)) % SIZE_2(gradFirst) + 3*{{intStride}}; // h-pos
	  
	  // round_off is a trick to enable integer division with ceil, even for negative numbers
	  // We use a large offset, for the inner part not to become negative.
	  const int round_off = ROUND_OFF;
	  const int round_off_s1 = {{intStride}} * round_off;
	  
	  // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
	  int xmin = (l - 3*{{intStride}} + round_off_s1 - 1) / {{intStride}} + 1 - round_off; // ceil (l - 3*{{intStride}}) / {{intStride}}
	  int ymin = (m - 3*{{intStride}} + round_off_s1 - 1) / {{intStride}} + 1 - round_off; // ceil (l - 3*{{intStride}}) / {{intStride}}
	  
	  // Same here:
	  int xmax = (l - 3*{{intStride}} + round_off_s1) / {{intStride}} - round_off; // floor (l - 3*{{intStride}}) / {{intStride}}
	  int ymax = (m - 3*{{intStride}} + round_off_s1) / {{intStride}} - round_off; // floor (m - 3*{{intStride}}) / {{intStride}}
	  
	  float sum = 0;
	  if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
	    xmin = max(0,xmin);
	    xmax = min(SIZE_3(gradOutput)-1,xmax);
	    
	    ymin = max(0,ymin);
	    ymax = min(SIZE_2(gradOutput)-1,ymax);
	    
	    for (int p = -3; p <= 3; p++) {
	      for (int o = -3; o <= 3; o++) {
	        // Get rbot1 data:
	        int s2o = {{intStride}} * o;
	        int s2p = {{intStride}} * p;
	        int idxbot1 = ((intSample * SIZE_1(rbot0) + (m+s2p)) * SIZE_2(rbot0) + (l+s2o)) * SIZE_3(rbot0) + n;
	        float bot1tmp = rbot1[idxbot1]; // rbot1[l+s2o,m+s2p,n]
	        
	        // Index offset for gradOutput in following loops:
	        int op = (p+3) * 7 + (o+3); // index[o,p]
	        int idxopoffset = (intSample * SIZE_1(gradOutput) + op);
	        
	        for (int y = ymin; y <= ymax; y++) {
	          for (int x = xmin; x <= xmax; x++) {
	            int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x; // gradOutput[x,y,o,p]
	            sum += gradOutput[idxgradOutput] * bot1tmp;
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = SIZE_1(gradFirst);
	  const int bot0index = ((n * SIZE_2(gradFirst)) + (m-3*{{intStride}})) * SIZE_3(gradFirst) + (l-3*{{intStride}});
	  gradFirst[bot0index + intSample*SIZE_1(gradFirst)*SIZE_2(gradFirst)*SIZE_3(gradFirst)] = sum / (float)sumelems;
	} }
'''

kernel_Correlation_updateGradSecond = '''
	#define ROUND_OFF 50000

	extern "C" __global__ void kernel_Correlation_updateGradSecond(
	  const int n,
	  const int intSample,
	  const float* rbot0,
	  const float* rbot1,
	  const float* gradOutput,
	  float* gradFirst,
	  float* gradSecond
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	  int n = intIndex % SIZE_1(gradSecond); // channels
	  int l = (intIndex / SIZE_1(gradSecond)) % SIZE_3(gradSecond) + 3*{{intStride}}; // w-pos
	  int m = (intIndex / SIZE_1(gradSecond) / SIZE_3(gradSecond)) % SIZE_2(gradSecond) + 3*{{intStride}}; // h-pos
	  
	  // round_off is a trick to enable integer division with ceil, even for negative numbers
	  // We use a large offset, for the inner part not to become negative.
	  const int round_off = ROUND_OFF;
	  const int round_off_s1 = {{intStride}} * round_off;
	  
	  float sum = 0;
	  for (int p = -3; p <= 3; p++) {
	    for (int o = -3; o <= 3; o++) {
	      int s2o = {{intStride}} * o;
	      int s2p = {{intStride}} * p;
	      
	      //Get X,Y ranges and clamp
	      // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
	      int xmin = (l - 3*{{intStride}} - s2o + round_off_s1 - 1) / {{intStride}} + 1 - round_off; // ceil (l - 3*{{intStride}} - s2o) / {{intStride}}
	      int ymin = (m - 3*{{intStride}} - s2p + round_off_s1 - 1) / {{intStride}} + 1 - round_off; // ceil (l - 3*{{intStride}} - s2o) / {{intStride}}
	      
	      // Same here:
	      int xmax = (l - 3*{{intStride}} - s2o + round_off_s1) / {{intStride}} - round_off; // floor (l - 3*{{intStride}} - s2o) / {{intStride}}
	      int ymax = (m - 3*{{intStride}} - s2p + round_off_s1) / {{intStride}} - round_off; // floor (m - 3*{{intStride}} - s2p) / {{intStride}}
          
	      if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
	        xmin = max(0,xmin);
	        xmax = min(SIZE_3(gradOutput)-1,xmax);
	        
	        ymin = max(0,ymin);
	        ymax = min(SIZE_2(gradOutput)-1,ymax);
	        
	        // Get rbot0 data:
	        int idxbot0 = ((intSample * SIZE_1(rbot0) + (m-s2p)) * SIZE_2(rbot0) + (l-s2o)) * SIZE_3(rbot0) + n;
	        float bot0tmp = rbot0[idxbot0]; // rbot1[l+s2o,m+s2p,n]
	        
	        // Index offset for gradOutput in following loops:
	        int op = (p+3) * 7 + (o+3); // index[o,p]
	        int idxopoffset = (intSample * SIZE_1(gradOutput) + op);
	        
	        for (int y = ymin; y <= ymax; y++) {
	          for (int x = xmin; x <= xmax; x++) {
	            int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x; // gradOutput[x,y,o,p]
	            sum += gradOutput[idxgradOutput] * bot0tmp;
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = SIZE_1(gradSecond);
	  const int bot1index = ((n * SIZE_2(gradSecond)) + (m-3*{{intStride}})) * SIZE_3(gradSecond) + (l-3*{{intStride}});
	  gradSecond[bot1index + intSample*SIZE_1(gradSecond)*SIZE_2(gradSecond)*SIZE_3(gradSecond)] = sum / (float)sumelems;
	} }
'''

def cupy_kernel(strFunction, objectVariables):
	strKernel = globals()[strFunction].replace('{{intStride}}', str(objectVariables['intStride']))

	while True:
		objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArg = int(objectMatch.group(2))

		strTensor = objectMatch.group(4)
		intSizes = objectVariables[strTensor].size()

		strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
	# end

	while True:
		objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArgs = int(objectMatch.group(2))
		strArgs = objectMatch.group(4).split(',')

		strTensor = strArgs[0]
		intStrides = objectVariables[strTensor].stride()
		strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

		strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
	# end

	return strKernel
# end

@cupy.util.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
	return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
# end

class _FunctionCorrelation(torch.autograd.Function):
	@staticmethod
	def forward(self, first, second, intStride):
		rbot0 = first.new_zeros([ first.size(0), first.size(2) + (6 * intStride), first.size(3) + (6 * intStride), first.size(1) ])
		rbot1 = first.new_zeros([ first.size(0), first.size(2) + (6 * intStride), first.size(3) + (6 * intStride), first.size(1) ])

		self.save_for_backward(first, second, rbot0, rbot1)

		self.intStride = intStride

		assert(first.is_contiguous() == True)
		assert(second.is_contiguous() == True)

		output = first.new_zeros([ first.size(0), 49, int(math.ceil(first.size(2) / intStride)), int(math.ceil(first.size(3) / intStride)) ])

		if first.is_cuda == True:
			n = first.size(2) * first.size(3)
			cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
				'intStride': self.intStride,
				'input': first,
				'output': rbot0
			}))(
				grid=tuple([ int((n + 16 - 1) / 16), first.size(1), first.size(0) ]),
				block=tuple([ 16, 1, 1 ]),
				args=[ n, first.data_ptr(), rbot0.data_ptr() ],
				stream=Stream
			)

			n = second.size(2) * second.size(3)
			cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
				'intStride': self.intStride,
				'input': second,
				'output': rbot1
			}))(
				grid=tuple([ int((n + 16 - 1) / 16), second.size(1), second.size(0) ]),
				block=tuple([ 16, 1, 1 ]),
				args=[ n, second.data_ptr(), rbot1.data_ptr() ],
				stream=Stream
			)

			n = output.size(1) * output.size(2) * output.size(3)
			cupy_launch('kernel_Correlation_updateOutput', cupy_kernel('kernel_Correlation_updateOutput', {
				'intStride': self.intStride,
				'rbot0': rbot0,
				'rbot1': rbot1,
				'top': output
			}))(
				grid=tuple([ output.size(3), output.size(2), output.size(0) ]),
				block=tuple([ 32, 1, 1 ]),
				shared_mem=first.size(1) * 4,
				args=[ n, rbot0.data_ptr(), rbot1.data_ptr(), output.data_ptr() ],
				stream=Stream
			)

		elif first.is_cuda == False:
			raise NotImplementedError()

		# end

		return output
	# end

	@staticmethod
	def backward(self, gradOutput):
		first, second, rbot0, rbot1 = self.saved_tensors

		assert(gradOutput.is_contiguous() == True)

		gradFirst = first.new_zeros([ first.size(0), first.size(1), first.size(2), first.size(3) ]) if self.needs_input_grad[0] == True else None
		gradSecond = first.new_zeros([ first.size(0), first.size(1), first.size(2), first.size(3) ]) if self.needs_input_grad[1] == True else None

		if first.is_cuda == True:
			if gradFirst is not None:
				for intSample in range(first.size(0)):
					n = first.size(1) * first.size(2) * first.size(3)
					cupy_launch('kernel_Correlation_updateGradFirst', cupy_kernel('kernel_Correlation_updateGradFirst', {
						'intStride': self.intStride,
						'rbot0': rbot0,
						'rbot1': rbot1,
						'gradOutput': gradOutput,
						'gradFirst': gradFirst,
						'gradSecond': None
					}))(
						grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
						block=tuple([ 512, 1, 1 ]),
						args=[ n, intSample, rbot0.data_ptr(), rbot1.data_ptr(), gradOutput.data_ptr(), gradFirst.data_ptr(), None ],
						stream=Stream
					)
				# end
			# end

			if gradSecond is not None:
				for intSample in range(first.size(0)):
					n = first.size(1) * first.size(2) * first.size(3)
					cupy_launch('kernel_Correlation_updateGradSecond', cupy_kernel('kernel_Correlation_updateGradSecond', {
						'intStride': self.intStride,
						'rbot0': rbot0,
						'rbot1': rbot1,
						'gradOutput': gradOutput,
						'gradFirst': None,
						'gradSecond': gradSecond
					}))(
						grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
						block=tuple([ 512, 1, 1 ]),
						args=[ n, intSample, rbot0.data_ptr(), rbot1.data_ptr(), gradOutput.data_ptr(), None, gradSecond.data_ptr() ],
						stream=Stream
					)
				# end
			# end

		elif first.is_cuda == False:
			raise NotImplementedError()

		# end

		return gradFirst, gradSecond, None
	# end
# end

def FunctionCorrelation(tensorFirst, tensorSecond, intStride):
	return _FunctionCorrelation.apply(tensorFirst, tensorSecond, intStride)
# end

class ModuleCorrelation(torch.nn.Module):
	def __init__(self):
		super(ModuleCorrelation, self).__init__()
	# end

	def forward(self, tensorFirst, tensorSecond, intStride):
		return _FunctionCorrelation.apply(tensorFirst, tensorSecond, intStride)
	# end
# end