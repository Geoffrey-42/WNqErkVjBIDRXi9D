фо*
І
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ћ
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%Зб8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02unknown8зў$

"Adam/Bottleneck2/shortbatch/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/Bottleneck2/shortbatch/beta/v

6Adam/Bottleneck2/shortbatch/beta/v/Read/ReadVariableOpReadVariableOp"Adam/Bottleneck2/shortbatch/beta/v*
_output_shapes
:*
dtype0

#Adam/Bottleneck2/shortbatch/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/Bottleneck2/shortbatch/gamma/v

7Adam/Bottleneck2/shortbatch/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/Bottleneck2/shortbatch/gamma/v*
_output_shapes
:*
dtype0

!Adam/Bottleneck2/projbatch/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/Bottleneck2/projbatch/beta/v

5Adam/Bottleneck2/projbatch/beta/v/Read/ReadVariableOpReadVariableOp!Adam/Bottleneck2/projbatch/beta/v*
_output_shapes
:*
dtype0

"Adam/Bottleneck2/projbatch/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/Bottleneck2/projbatch/gamma/v

6Adam/Bottleneck2/projbatch/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/Bottleneck2/projbatch/gamma/v*
_output_shapes
:*
dtype0

 Adam/Bottleneck2/projconv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/Bottleneck2/projconv/bias/v

4Adam/Bottleneck2/projconv/bias/v/Read/ReadVariableOpReadVariableOp Adam/Bottleneck2/projconv/bias/v*
_output_shapes
:*
dtype0
Ј
"Adam/Bottleneck2/projconv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*3
shared_name$"Adam/Bottleneck2/projconv/kernel/v
Ё
6Adam/Bottleneck2/projconv/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/Bottleneck2/projconv/kernel/v*&
_output_shapes
:0*
dtype0

"Adam/Bottleneck2/depthbatch/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*3
shared_name$"Adam/Bottleneck2/depthbatch/beta/v

6Adam/Bottleneck2/depthbatch/beta/v/Read/ReadVariableOpReadVariableOp"Adam/Bottleneck2/depthbatch/beta/v*
_output_shapes
:0*
dtype0

#Adam/Bottleneck2/depthbatch/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*4
shared_name%#Adam/Bottleneck2/depthbatch/gamma/v

7Adam/Bottleneck2/depthbatch/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/Bottleneck2/depthbatch/gamma/v*
_output_shapes
:0*
dtype0

!Adam/Bottleneck2/depthconv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!Adam/Bottleneck2/depthconv/bias/v

5Adam/Bottleneck2/depthconv/bias/v/Read/ReadVariableOpReadVariableOp!Adam/Bottleneck2/depthconv/bias/v*
_output_shapes
:0*
dtype0
О
-Adam/Bottleneck2/depthconv/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*>
shared_name/-Adam/Bottleneck2/depthconv/depthwise_kernel/v
З
AAdam/Bottleneck2/depthconv/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp-Adam/Bottleneck2/depthconv/depthwise_kernel/v*&
_output_shapes
:0*
dtype0

 Adam/Bottleneck2/expbatch/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*1
shared_name" Adam/Bottleneck2/expbatch/beta/v

4Adam/Bottleneck2/expbatch/beta/v/Read/ReadVariableOpReadVariableOp Adam/Bottleneck2/expbatch/beta/v*
_output_shapes
:0*
dtype0

!Adam/Bottleneck2/expbatch/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!Adam/Bottleneck2/expbatch/gamma/v

5Adam/Bottleneck2/expbatch/gamma/v/Read/ReadVariableOpReadVariableOp!Adam/Bottleneck2/expbatch/gamma/v*
_output_shapes
:0*
dtype0

Adam/Bottleneck2/expconv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*0
shared_name!Adam/Bottleneck2/expconv/bias/v

3Adam/Bottleneck2/expconv/bias/v/Read/ReadVariableOpReadVariableOpAdam/Bottleneck2/expconv/bias/v*
_output_shapes
:0*
dtype0
І
!Adam/Bottleneck2/expconv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!Adam/Bottleneck2/expconv/kernel/v

5Adam/Bottleneck2/expconv/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/Bottleneck2/expconv/kernel/v*&
_output_shapes
:0*
dtype0

"Adam/Bottleneck1/shortbatch/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/Bottleneck1/shortbatch/beta/v

6Adam/Bottleneck1/shortbatch/beta/v/Read/ReadVariableOpReadVariableOp"Adam/Bottleneck1/shortbatch/beta/v*
_output_shapes
:*
dtype0

#Adam/Bottleneck1/shortbatch/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/Bottleneck1/shortbatch/gamma/v

7Adam/Bottleneck1/shortbatch/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/Bottleneck1/shortbatch/gamma/v*
_output_shapes
:*
dtype0

!Adam/Bottleneck1/projbatch/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/Bottleneck1/projbatch/beta/v

5Adam/Bottleneck1/projbatch/beta/v/Read/ReadVariableOpReadVariableOp!Adam/Bottleneck1/projbatch/beta/v*
_output_shapes
:*
dtype0

"Adam/Bottleneck1/projbatch/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/Bottleneck1/projbatch/gamma/v

6Adam/Bottleneck1/projbatch/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/Bottleneck1/projbatch/gamma/v*
_output_shapes
:*
dtype0

 Adam/Bottleneck1/projconv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/Bottleneck1/projconv/bias/v

4Adam/Bottleneck1/projconv/bias/v/Read/ReadVariableOpReadVariableOp Adam/Bottleneck1/projconv/bias/v*
_output_shapes
:*
dtype0
Ј
"Adam/Bottleneck1/projconv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/Bottleneck1/projconv/kernel/v
Ё
6Adam/Bottleneck1/projconv/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/Bottleneck1/projconv/kernel/v*&
_output_shapes
:*
dtype0

"Adam/Bottleneck1/depthbatch/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/Bottleneck1/depthbatch/beta/v

6Adam/Bottleneck1/depthbatch/beta/v/Read/ReadVariableOpReadVariableOp"Adam/Bottleneck1/depthbatch/beta/v*
_output_shapes
:*
dtype0

#Adam/Bottleneck1/depthbatch/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/Bottleneck1/depthbatch/gamma/v

7Adam/Bottleneck1/depthbatch/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/Bottleneck1/depthbatch/gamma/v*
_output_shapes
:*
dtype0

!Adam/Bottleneck1/depthconv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/Bottleneck1/depthconv/bias/v

5Adam/Bottleneck1/depthconv/bias/v/Read/ReadVariableOpReadVariableOp!Adam/Bottleneck1/depthconv/bias/v*
_output_shapes
:*
dtype0
О
-Adam/Bottleneck1/depthconv/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/Bottleneck1/depthconv/depthwise_kernel/v
З
AAdam/Bottleneck1/depthconv/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp-Adam/Bottleneck1/depthconv/depthwise_kernel/v*&
_output_shapes
:*
dtype0

 Adam/Bottleneck1/expbatch/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/Bottleneck1/expbatch/beta/v

4Adam/Bottleneck1/expbatch/beta/v/Read/ReadVariableOpReadVariableOp Adam/Bottleneck1/expbatch/beta/v*
_output_shapes
:*
dtype0

!Adam/Bottleneck1/expbatch/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/Bottleneck1/expbatch/gamma/v

5Adam/Bottleneck1/expbatch/gamma/v/Read/ReadVariableOpReadVariableOp!Adam/Bottleneck1/expbatch/gamma/v*
_output_shapes
:*
dtype0

Adam/Bottleneck1/expconv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/Bottleneck1/expconv/bias/v

3Adam/Bottleneck1/expconv/bias/v/Read/ReadVariableOpReadVariableOpAdam/Bottleneck1/expconv/bias/v*
_output_shapes
:*
dtype0
І
!Adam/Bottleneck1/expconv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/Bottleneck1/expconv/kernel/v

5Adam/Bottleneck1/expconv/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/Bottleneck1/expconv/kernel/v*&
_output_shapes
:*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes
:		*
dtype0

"Adam/batch_normalization_34/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_34/beta/v

6Adam/batch_normalization_34/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_34/beta/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_34/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_34/gamma/v

7Adam/batch_normalization_34/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_34/gamma/v*
_output_shapes
:*
dtype0

Adam/conv2d_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_18/bias/v
{
)Adam/conv2d_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_18/kernel/v

+Adam/conv2d_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/v*&
_output_shapes
:*
dtype0

"Adam/Bottleneck2/shortbatch/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/Bottleneck2/shortbatch/beta/m

6Adam/Bottleneck2/shortbatch/beta/m/Read/ReadVariableOpReadVariableOp"Adam/Bottleneck2/shortbatch/beta/m*
_output_shapes
:*
dtype0

#Adam/Bottleneck2/shortbatch/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/Bottleneck2/shortbatch/gamma/m

7Adam/Bottleneck2/shortbatch/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/Bottleneck2/shortbatch/gamma/m*
_output_shapes
:*
dtype0

!Adam/Bottleneck2/projbatch/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/Bottleneck2/projbatch/beta/m

5Adam/Bottleneck2/projbatch/beta/m/Read/ReadVariableOpReadVariableOp!Adam/Bottleneck2/projbatch/beta/m*
_output_shapes
:*
dtype0

"Adam/Bottleneck2/projbatch/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/Bottleneck2/projbatch/gamma/m

6Adam/Bottleneck2/projbatch/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/Bottleneck2/projbatch/gamma/m*
_output_shapes
:*
dtype0

 Adam/Bottleneck2/projconv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/Bottleneck2/projconv/bias/m

4Adam/Bottleneck2/projconv/bias/m/Read/ReadVariableOpReadVariableOp Adam/Bottleneck2/projconv/bias/m*
_output_shapes
:*
dtype0
Ј
"Adam/Bottleneck2/projconv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*3
shared_name$"Adam/Bottleneck2/projconv/kernel/m
Ё
6Adam/Bottleneck2/projconv/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/Bottleneck2/projconv/kernel/m*&
_output_shapes
:0*
dtype0

"Adam/Bottleneck2/depthbatch/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*3
shared_name$"Adam/Bottleneck2/depthbatch/beta/m

6Adam/Bottleneck2/depthbatch/beta/m/Read/ReadVariableOpReadVariableOp"Adam/Bottleneck2/depthbatch/beta/m*
_output_shapes
:0*
dtype0

#Adam/Bottleneck2/depthbatch/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*4
shared_name%#Adam/Bottleneck2/depthbatch/gamma/m

7Adam/Bottleneck2/depthbatch/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/Bottleneck2/depthbatch/gamma/m*
_output_shapes
:0*
dtype0

!Adam/Bottleneck2/depthconv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!Adam/Bottleneck2/depthconv/bias/m

5Adam/Bottleneck2/depthconv/bias/m/Read/ReadVariableOpReadVariableOp!Adam/Bottleneck2/depthconv/bias/m*
_output_shapes
:0*
dtype0
О
-Adam/Bottleneck2/depthconv/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*>
shared_name/-Adam/Bottleneck2/depthconv/depthwise_kernel/m
З
AAdam/Bottleneck2/depthconv/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp-Adam/Bottleneck2/depthconv/depthwise_kernel/m*&
_output_shapes
:0*
dtype0

 Adam/Bottleneck2/expbatch/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*1
shared_name" Adam/Bottleneck2/expbatch/beta/m

4Adam/Bottleneck2/expbatch/beta/m/Read/ReadVariableOpReadVariableOp Adam/Bottleneck2/expbatch/beta/m*
_output_shapes
:0*
dtype0

!Adam/Bottleneck2/expbatch/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!Adam/Bottleneck2/expbatch/gamma/m

5Adam/Bottleneck2/expbatch/gamma/m/Read/ReadVariableOpReadVariableOp!Adam/Bottleneck2/expbatch/gamma/m*
_output_shapes
:0*
dtype0

Adam/Bottleneck2/expconv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*0
shared_name!Adam/Bottleneck2/expconv/bias/m

3Adam/Bottleneck2/expconv/bias/m/Read/ReadVariableOpReadVariableOpAdam/Bottleneck2/expconv/bias/m*
_output_shapes
:0*
dtype0
І
!Adam/Bottleneck2/expconv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!Adam/Bottleneck2/expconv/kernel/m

5Adam/Bottleneck2/expconv/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/Bottleneck2/expconv/kernel/m*&
_output_shapes
:0*
dtype0

"Adam/Bottleneck1/shortbatch/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/Bottleneck1/shortbatch/beta/m

6Adam/Bottleneck1/shortbatch/beta/m/Read/ReadVariableOpReadVariableOp"Adam/Bottleneck1/shortbatch/beta/m*
_output_shapes
:*
dtype0

#Adam/Bottleneck1/shortbatch/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/Bottleneck1/shortbatch/gamma/m

7Adam/Bottleneck1/shortbatch/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/Bottleneck1/shortbatch/gamma/m*
_output_shapes
:*
dtype0

!Adam/Bottleneck1/projbatch/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/Bottleneck1/projbatch/beta/m

5Adam/Bottleneck1/projbatch/beta/m/Read/ReadVariableOpReadVariableOp!Adam/Bottleneck1/projbatch/beta/m*
_output_shapes
:*
dtype0

"Adam/Bottleneck1/projbatch/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/Bottleneck1/projbatch/gamma/m

6Adam/Bottleneck1/projbatch/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/Bottleneck1/projbatch/gamma/m*
_output_shapes
:*
dtype0

 Adam/Bottleneck1/projconv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/Bottleneck1/projconv/bias/m

4Adam/Bottleneck1/projconv/bias/m/Read/ReadVariableOpReadVariableOp Adam/Bottleneck1/projconv/bias/m*
_output_shapes
:*
dtype0
Ј
"Adam/Bottleneck1/projconv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/Bottleneck1/projconv/kernel/m
Ё
6Adam/Bottleneck1/projconv/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/Bottleneck1/projconv/kernel/m*&
_output_shapes
:*
dtype0

"Adam/Bottleneck1/depthbatch/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/Bottleneck1/depthbatch/beta/m

6Adam/Bottleneck1/depthbatch/beta/m/Read/ReadVariableOpReadVariableOp"Adam/Bottleneck1/depthbatch/beta/m*
_output_shapes
:*
dtype0

#Adam/Bottleneck1/depthbatch/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/Bottleneck1/depthbatch/gamma/m

7Adam/Bottleneck1/depthbatch/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/Bottleneck1/depthbatch/gamma/m*
_output_shapes
:*
dtype0

!Adam/Bottleneck1/depthconv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/Bottleneck1/depthconv/bias/m

5Adam/Bottleneck1/depthconv/bias/m/Read/ReadVariableOpReadVariableOp!Adam/Bottleneck1/depthconv/bias/m*
_output_shapes
:*
dtype0
О
-Adam/Bottleneck1/depthconv/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/Bottleneck1/depthconv/depthwise_kernel/m
З
AAdam/Bottleneck1/depthconv/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp-Adam/Bottleneck1/depthconv/depthwise_kernel/m*&
_output_shapes
:*
dtype0

 Adam/Bottleneck1/expbatch/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/Bottleneck1/expbatch/beta/m

4Adam/Bottleneck1/expbatch/beta/m/Read/ReadVariableOpReadVariableOp Adam/Bottleneck1/expbatch/beta/m*
_output_shapes
:*
dtype0

!Adam/Bottleneck1/expbatch/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/Bottleneck1/expbatch/gamma/m

5Adam/Bottleneck1/expbatch/gamma/m/Read/ReadVariableOpReadVariableOp!Adam/Bottleneck1/expbatch/gamma/m*
_output_shapes
:*
dtype0

Adam/Bottleneck1/expconv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/Bottleneck1/expconv/bias/m

3Adam/Bottleneck1/expconv/bias/m/Read/ReadVariableOpReadVariableOpAdam/Bottleneck1/expconv/bias/m*
_output_shapes
:*
dtype0
І
!Adam/Bottleneck1/expconv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/Bottleneck1/expconv/kernel/m

5Adam/Bottleneck1/expconv/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/Bottleneck1/expconv/kernel/m*&
_output_shapes
:*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes
:		*
dtype0

"Adam/batch_normalization_34/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_34/beta/m

6Adam/batch_normalization_34/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_34/beta/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_34/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_34/gamma/m

7Adam/batch_normalization_34/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_34/gamma/m*
_output_shapes
:*
dtype0

Adam/conv2d_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_18/bias/m
{
)Adam/conv2d_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_18/kernel/m

+Adam/conv2d_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/m*&
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
Є
&Bottleneck2/shortbatch/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Bottleneck2/shortbatch/moving_variance

:Bottleneck2/shortbatch/moving_variance/Read/ReadVariableOpReadVariableOp&Bottleneck2/shortbatch/moving_variance*
_output_shapes
:*
dtype0

"Bottleneck2/shortbatch/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Bottleneck2/shortbatch/moving_mean

6Bottleneck2/shortbatch/moving_mean/Read/ReadVariableOpReadVariableOp"Bottleneck2/shortbatch/moving_mean*
_output_shapes
:*
dtype0
Ђ
%Bottleneck2/projbatch/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Bottleneck2/projbatch/moving_variance

9Bottleneck2/projbatch/moving_variance/Read/ReadVariableOpReadVariableOp%Bottleneck2/projbatch/moving_variance*
_output_shapes
:*
dtype0

!Bottleneck2/projbatch/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Bottleneck2/projbatch/moving_mean

5Bottleneck2/projbatch/moving_mean/Read/ReadVariableOpReadVariableOp!Bottleneck2/projbatch/moving_mean*
_output_shapes
:*
dtype0
Є
&Bottleneck2/depthbatch/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*7
shared_name(&Bottleneck2/depthbatch/moving_variance

:Bottleneck2/depthbatch/moving_variance/Read/ReadVariableOpReadVariableOp&Bottleneck2/depthbatch/moving_variance*
_output_shapes
:0*
dtype0

"Bottleneck2/depthbatch/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*3
shared_name$"Bottleneck2/depthbatch/moving_mean

6Bottleneck2/depthbatch/moving_mean/Read/ReadVariableOpReadVariableOp"Bottleneck2/depthbatch/moving_mean*
_output_shapes
:0*
dtype0
 
$Bottleneck2/expbatch/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*5
shared_name&$Bottleneck2/expbatch/moving_variance

8Bottleneck2/expbatch/moving_variance/Read/ReadVariableOpReadVariableOp$Bottleneck2/expbatch/moving_variance*
_output_shapes
:0*
dtype0

 Bottleneck2/expbatch/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*1
shared_name" Bottleneck2/expbatch/moving_mean

4Bottleneck2/expbatch/moving_mean/Read/ReadVariableOpReadVariableOp Bottleneck2/expbatch/moving_mean*
_output_shapes
:0*
dtype0

Bottleneck2/shortbatch/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameBottleneck2/shortbatch/beta

/Bottleneck2/shortbatch/beta/Read/ReadVariableOpReadVariableOpBottleneck2/shortbatch/beta*
_output_shapes
:*
dtype0

Bottleneck2/shortbatch/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameBottleneck2/shortbatch/gamma

0Bottleneck2/shortbatch/gamma/Read/ReadVariableOpReadVariableOpBottleneck2/shortbatch/gamma*
_output_shapes
:*
dtype0

Bottleneck2/projbatch/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameBottleneck2/projbatch/beta

.Bottleneck2/projbatch/beta/Read/ReadVariableOpReadVariableOpBottleneck2/projbatch/beta*
_output_shapes
:*
dtype0

Bottleneck2/projbatch/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameBottleneck2/projbatch/gamma

/Bottleneck2/projbatch/gamma/Read/ReadVariableOpReadVariableOpBottleneck2/projbatch/gamma*
_output_shapes
:*
dtype0

Bottleneck2/projconv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameBottleneck2/projconv/bias

-Bottleneck2/projconv/bias/Read/ReadVariableOpReadVariableOpBottleneck2/projconv/bias*
_output_shapes
:*
dtype0

Bottleneck2/projconv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*,
shared_nameBottleneck2/projconv/kernel

/Bottleneck2/projconv/kernel/Read/ReadVariableOpReadVariableOpBottleneck2/projconv/kernel*&
_output_shapes
:0*
dtype0

Bottleneck2/depthbatch/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*,
shared_nameBottleneck2/depthbatch/beta

/Bottleneck2/depthbatch/beta/Read/ReadVariableOpReadVariableOpBottleneck2/depthbatch/beta*
_output_shapes
:0*
dtype0

Bottleneck2/depthbatch/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*-
shared_nameBottleneck2/depthbatch/gamma

0Bottleneck2/depthbatch/gamma/Read/ReadVariableOpReadVariableOpBottleneck2/depthbatch/gamma*
_output_shapes
:0*
dtype0

Bottleneck2/depthconv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_nameBottleneck2/depthconv/bias

.Bottleneck2/depthconv/bias/Read/ReadVariableOpReadVariableOpBottleneck2/depthconv/bias*
_output_shapes
:0*
dtype0
А
&Bottleneck2/depthconv/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*7
shared_name(&Bottleneck2/depthconv/depthwise_kernel
Љ
:Bottleneck2/depthconv/depthwise_kernel/Read/ReadVariableOpReadVariableOp&Bottleneck2/depthconv/depthwise_kernel*&
_output_shapes
:0*
dtype0

Bottleneck2/expbatch/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0**
shared_nameBottleneck2/expbatch/beta

-Bottleneck2/expbatch/beta/Read/ReadVariableOpReadVariableOpBottleneck2/expbatch/beta*
_output_shapes
:0*
dtype0

Bottleneck2/expbatch/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_nameBottleneck2/expbatch/gamma

.Bottleneck2/expbatch/gamma/Read/ReadVariableOpReadVariableOpBottleneck2/expbatch/gamma*
_output_shapes
:0*
dtype0

Bottleneck2/expconv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*)
shared_nameBottleneck2/expconv/bias

,Bottleneck2/expconv/bias/Read/ReadVariableOpReadVariableOpBottleneck2/expconv/bias*
_output_shapes
:0*
dtype0

Bottleneck2/expconv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_nameBottleneck2/expconv/kernel

.Bottleneck2/expconv/kernel/Read/ReadVariableOpReadVariableOpBottleneck2/expconv/kernel*&
_output_shapes
:0*
dtype0
Є
&Bottleneck1/shortbatch/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Bottleneck1/shortbatch/moving_variance

:Bottleneck1/shortbatch/moving_variance/Read/ReadVariableOpReadVariableOp&Bottleneck1/shortbatch/moving_variance*
_output_shapes
:*
dtype0

"Bottleneck1/shortbatch/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Bottleneck1/shortbatch/moving_mean

6Bottleneck1/shortbatch/moving_mean/Read/ReadVariableOpReadVariableOp"Bottleneck1/shortbatch/moving_mean*
_output_shapes
:*
dtype0
Ђ
%Bottleneck1/projbatch/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Bottleneck1/projbatch/moving_variance

9Bottleneck1/projbatch/moving_variance/Read/ReadVariableOpReadVariableOp%Bottleneck1/projbatch/moving_variance*
_output_shapes
:*
dtype0

!Bottleneck1/projbatch/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Bottleneck1/projbatch/moving_mean

5Bottleneck1/projbatch/moving_mean/Read/ReadVariableOpReadVariableOp!Bottleneck1/projbatch/moving_mean*
_output_shapes
:*
dtype0
Є
&Bottleneck1/depthbatch/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Bottleneck1/depthbatch/moving_variance

:Bottleneck1/depthbatch/moving_variance/Read/ReadVariableOpReadVariableOp&Bottleneck1/depthbatch/moving_variance*
_output_shapes
:*
dtype0

"Bottleneck1/depthbatch/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Bottleneck1/depthbatch/moving_mean

6Bottleneck1/depthbatch/moving_mean/Read/ReadVariableOpReadVariableOp"Bottleneck1/depthbatch/moving_mean*
_output_shapes
:*
dtype0
 
$Bottleneck1/expbatch/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Bottleneck1/expbatch/moving_variance

8Bottleneck1/expbatch/moving_variance/Read/ReadVariableOpReadVariableOp$Bottleneck1/expbatch/moving_variance*
_output_shapes
:*
dtype0

 Bottleneck1/expbatch/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Bottleneck1/expbatch/moving_mean

4Bottleneck1/expbatch/moving_mean/Read/ReadVariableOpReadVariableOp Bottleneck1/expbatch/moving_mean*
_output_shapes
:*
dtype0

Bottleneck1/shortbatch/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameBottleneck1/shortbatch/beta

/Bottleneck1/shortbatch/beta/Read/ReadVariableOpReadVariableOpBottleneck1/shortbatch/beta*
_output_shapes
:*
dtype0

Bottleneck1/shortbatch/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameBottleneck1/shortbatch/gamma

0Bottleneck1/shortbatch/gamma/Read/ReadVariableOpReadVariableOpBottleneck1/shortbatch/gamma*
_output_shapes
:*
dtype0

Bottleneck1/projbatch/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameBottleneck1/projbatch/beta

.Bottleneck1/projbatch/beta/Read/ReadVariableOpReadVariableOpBottleneck1/projbatch/beta*
_output_shapes
:*
dtype0

Bottleneck1/projbatch/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameBottleneck1/projbatch/gamma

/Bottleneck1/projbatch/gamma/Read/ReadVariableOpReadVariableOpBottleneck1/projbatch/gamma*
_output_shapes
:*
dtype0

Bottleneck1/projconv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameBottleneck1/projconv/bias

-Bottleneck1/projconv/bias/Read/ReadVariableOpReadVariableOpBottleneck1/projconv/bias*
_output_shapes
:*
dtype0

Bottleneck1/projconv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameBottleneck1/projconv/kernel

/Bottleneck1/projconv/kernel/Read/ReadVariableOpReadVariableOpBottleneck1/projconv/kernel*&
_output_shapes
:*
dtype0

Bottleneck1/depthbatch/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameBottleneck1/depthbatch/beta

/Bottleneck1/depthbatch/beta/Read/ReadVariableOpReadVariableOpBottleneck1/depthbatch/beta*
_output_shapes
:*
dtype0

Bottleneck1/depthbatch/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameBottleneck1/depthbatch/gamma

0Bottleneck1/depthbatch/gamma/Read/ReadVariableOpReadVariableOpBottleneck1/depthbatch/gamma*
_output_shapes
:*
dtype0

Bottleneck1/depthconv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameBottleneck1/depthconv/bias

.Bottleneck1/depthconv/bias/Read/ReadVariableOpReadVariableOpBottleneck1/depthconv/bias*
_output_shapes
:*
dtype0
А
&Bottleneck1/depthconv/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Bottleneck1/depthconv/depthwise_kernel
Љ
:Bottleneck1/depthconv/depthwise_kernel/Read/ReadVariableOpReadVariableOp&Bottleneck1/depthconv/depthwise_kernel*&
_output_shapes
:*
dtype0

Bottleneck1/expbatch/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameBottleneck1/expbatch/beta

-Bottleneck1/expbatch/beta/Read/ReadVariableOpReadVariableOpBottleneck1/expbatch/beta*
_output_shapes
:*
dtype0

Bottleneck1/expbatch/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameBottleneck1/expbatch/gamma

.Bottleneck1/expbatch/gamma/Read/ReadVariableOpReadVariableOpBottleneck1/expbatch/gamma*
_output_shapes
:*
dtype0

Bottleneck1/expconv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameBottleneck1/expconv/bias

,Bottleneck1/expconv/bias/Read/ReadVariableOpReadVariableOpBottleneck1/expconv/bias*
_output_shapes
:*
dtype0

Bottleneck1/expconv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameBottleneck1/expconv/kernel

.Bottleneck1/expconv/kernel/Read/ReadVariableOpReadVariableOpBottleneck1/expconv/kernel*&
_output_shapes
:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:		*
dtype0
Є
&batch_normalization_34/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_34/moving_variance

:batch_normalization_34/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_34/moving_variance*
_output_shapes
:*
dtype0

"batch_normalization_34/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_34/moving_mean

6batch_normalization_34/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_34/moving_mean*
_output_shapes
:*
dtype0

batch_normalization_34/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_34/beta

/batch_normalization_34/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_34/beta*
_output_shapes
:*
dtype0

batch_normalization_34/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_34/gamma

0batch_normalization_34/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_34/gamma*
_output_shapes
:*
dtype0
t
conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_18/bias
m
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
_output_shapes
:*
dtype0

conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_18/kernel
}
$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*&
_output_shapes
:*
dtype0

serving_default_input_3Placeholder*1
_output_shapes
:џџџџџџџџџрр*
dtype0*&
shape:џџџџџџџџџрр
ф
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3conv2d_18/kernelconv2d_18/biasbatch_normalization_34/gammabatch_normalization_34/beta"batch_normalization_34/moving_mean&batch_normalization_34/moving_varianceBottleneck1/expconv/kernelBottleneck1/expconv/biasBottleneck1/expbatch/gammaBottleneck1/expbatch/beta Bottleneck1/expbatch/moving_mean$Bottleneck1/expbatch/moving_variance&Bottleneck1/depthconv/depthwise_kernelBottleneck1/depthconv/biasBottleneck1/depthbatch/gammaBottleneck1/depthbatch/beta"Bottleneck1/depthbatch/moving_mean&Bottleneck1/depthbatch/moving_varianceBottleneck1/projconv/kernelBottleneck1/projconv/biasBottleneck1/projbatch/gammaBottleneck1/projbatch/beta!Bottleneck1/projbatch/moving_mean%Bottleneck1/projbatch/moving_varianceBottleneck1/shortbatch/gammaBottleneck1/shortbatch/beta"Bottleneck1/shortbatch/moving_mean&Bottleneck1/shortbatch/moving_varianceBottleneck2/expconv/kernelBottleneck2/expconv/biasBottleneck2/expbatch/gammaBottleneck2/expbatch/beta Bottleneck2/expbatch/moving_mean$Bottleneck2/expbatch/moving_variance&Bottleneck2/depthconv/depthwise_kernelBottleneck2/depthconv/biasBottleneck2/depthbatch/gammaBottleneck2/depthbatch/beta"Bottleneck2/depthbatch/moving_mean&Bottleneck2/depthbatch/moving_varianceBottleneck2/projconv/kernelBottleneck2/projconv/biasBottleneck2/projbatch/gammaBottleneck2/projbatch/beta!Bottleneck2/projbatch/moving_mean%Bottleneck2/projbatch/moving_varianceBottleneck2/shortbatch/gammaBottleneck2/shortbatch/beta"Bottleneck2/shortbatch/moving_mean&Bottleneck2/shortbatch/moving_variancedense_2/kerneldense_2/bias*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_49546

NoOpNoOp
е
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB Bј
Є
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer-10
layer_with_weights-3
layer-11
layer-12
layer-13
layer-14
layer_with_weights-4
layer-15
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 

	keras_api* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses* 
Ш
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias
 )_jit_compiled_convolution_op*
е
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0axis
	1gamma
2beta
3moving_mean
4moving_variance*

5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses* 

;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses* 
Ѕ
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
G_random_generator* 
ї
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Nexpconv
Oexpbatch
P	depthconv
Q
depthbatch
Rprojconv
S	projbatch
T
shortbatch*

U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses* 
Ѕ
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
a_random_generator* 
ї
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
hexpconv
iexpbatch
j	depthconv
k
depthbatch
lprojconv
m	projbatch
n
shortbatch*

o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses* 
Ѕ
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
{_random_generator* 

|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
Ў
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
Ш
'0
(1
12
23
34
45
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
 28
Ё29
Ђ30
Ѓ31
Є32
Ѕ33
І34
Ї35
Ј36
Љ37
Њ38
Ћ39
Ќ40
­41
Ў42
Џ43
А44
Б45
В46
Г47
Д48
Е49
50
51*
Ј
'0
(1
12
23
4
5
6
7
8
9
10
11
12
13
14
15
16
17
 18
Ё19
Ђ20
Ѓ21
Є22
Ѕ23
І24
Ї25
Ј26
Љ27
Њ28
Ћ29
Ќ30
­31
32
33*
* 
Е
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
Лtrace_0
Мtrace_1
Нtrace_2
Оtrace_3* 
:
Пtrace_0
Рtrace_1
Сtrace_2
Тtrace_3* 
* 

	Уiter
Фbeta_1
Хbeta_2

Цdecay'm(m1m2m	m	m	m	m	m	m	m	m	m	m	m	m	m	m 	mЁ	mЂ	 mЃ	ЁmЄ	ЂmЅ	ЃmІ	ЄmЇ	ЅmЈ	ІmЉ	ЇmЊ	ЈmЋ	ЉmЌ	Њm­	ЋmЎ	ЌmЏ	­mА'vБ(vВ1vГ2vД	vЕ	vЖ	vЗ	vИ	vЙ	vК	vЛ	vМ	vН	vО	vП	vР	vС	vТ	vУ	vФ	 vХ	ЁvЦ	ЂvЧ	ЃvШ	ЄvЩ	ЅvЪ	ІvЫ	ЇvЬ	ЈvЭ	ЉvЮ	ЊvЯ	Ћvа	Ќvб	­vв*

Чserving_default* 
* 
* 
* 
* 

Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses* 

Эtrace_0* 

Юtrace_0* 

'0
(1*

'0
(1*
* 

Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

дtrace_0* 

еtrace_0* 
`Z
VARIABLE_VALUEconv2d_18/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_18/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
10
21
32
43*

10
21*
* 

жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

лtrace_0
мtrace_1* 

нtrace_0
оtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_34/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_34/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_34/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_34/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses* 

фtrace_0* 

хtrace_0* 
* 
* 
* 

цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses* 

ыtrace_0* 

ьtrace_0* 
* 
* 
* 

эnon_trainable_variables
юlayers
яmetrics
 №layer_regularization_losses
ёlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses* 

ђtrace_0
ѓtrace_1* 

єtrace_0
ѕtrace_1* 
* 
Р
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21*
x
0
1
2
3
4
5
6
7
8
9
10
11
12
13*
* 

іnon_trainable_variables
їlayers
јmetrics
 љlayer_regularization_losses
њlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

ћtrace_0
ќtrace_1* 

§trace_0
ўtrace_1* 
б
џ	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op*
р
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
moving_variance*
л
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
depthwise_kernel
	bias
!_jit_compiled_convolution_op*
р
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
moving_variance*
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses
kernel
	bias
!Ё_jit_compiled_convolution_op*
р
Ђ	variables
Ѓtrainable_variables
Єregularization_losses
Ѕ	keras_api
І__call__
+Ї&call_and_return_all_conditional_losses
	Јaxis

gamma
	beta
moving_mean
moving_variance*
р
Љ	variables
Њtrainable_variables
Ћregularization_losses
Ќ	keras_api
­__call__
+Ў&call_and_return_all_conditional_losses
	Џaxis

gamma
	beta
moving_mean
moving_variance*
* 
* 
* 

Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses* 

Еtrace_0* 

Жtrace_0* 
* 
* 
* 

Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 

Мtrace_0
Нtrace_1* 

Оtrace_0
Пtrace_1* 
* 
Р
 0
Ё1
Ђ2
Ѓ3
Є4
Ѕ5
І6
Ї7
Ј8
Љ9
Њ10
Ћ11
Ќ12
­13
Ў14
Џ15
А16
Б17
В18
Г19
Д20
Е21*
x
 0
Ё1
Ђ2
Ѓ3
Є4
Ѕ5
І6
Ї7
Ј8
Љ9
Њ10
Ћ11
Ќ12
­13*
* 

Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

Хtrace_0
Цtrace_1* 

Чtrace_0
Шtrace_1* 
б
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses
 kernel
	Ёbias
!Я_jit_compiled_convolution_op*
р
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses
	жaxis

Ђgamma
	Ѓbeta
Ўmoving_mean
Џmoving_variance*
л
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
л__call__
+м&call_and_return_all_conditional_losses
Єdepthwise_kernel
	Ѕbias
!н_jit_compiled_convolution_op*
р
о	variables
пtrainable_variables
рregularization_losses
с	keras_api
т__call__
+у&call_and_return_all_conditional_losses
	фaxis

Іgamma
	Їbeta
Аmoving_mean
Бmoving_variance*
б
х	variables
цtrainable_variables
чregularization_losses
ш	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses
Јkernel
	Љbias
!ы_jit_compiled_convolution_op*
р
ь	variables
эtrainable_variables
юregularization_losses
я	keras_api
№__call__
+ё&call_and_return_all_conditional_losses
	ђaxis

Њgamma
	Ћbeta
Вmoving_mean
Гmoving_variance*
р
ѓ	variables
єtrainable_variables
ѕregularization_losses
і	keras_api
ї__call__
+ј&call_and_return_all_conditional_losses
	љaxis

Ќgamma
	­beta
Дmoving_mean
Еmoving_variance*
* 
* 
* 

њnon_trainable_variables
ћlayers
ќmetrics
 §layer_regularization_losses
ўlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses* 

џtrace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEBottleneck1/expconv/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEBottleneck1/expconv/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEBottleneck1/expbatch/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEBottleneck1/expbatch/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&Bottleneck1/depthconv/depthwise_kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEBottleneck1/depthconv/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEBottleneck1/depthbatch/gamma'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEBottleneck1/depthbatch/beta'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEBottleneck1/projconv/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEBottleneck1/projconv/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEBottleneck1/projbatch/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEBottleneck1/projbatch/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEBottleneck1/shortbatch/gamma'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEBottleneck1/shortbatch/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE Bottleneck1/expbatch/moving_mean'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$Bottleneck1/expbatch/moving_variance'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"Bottleneck1/depthbatch/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&Bottleneck1/depthbatch/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!Bottleneck1/projbatch/moving_mean'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%Bottleneck1/projbatch/moving_variance'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"Bottleneck1/shortbatch/moving_mean'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&Bottleneck1/shortbatch/moving_variance'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEBottleneck2/expconv/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEBottleneck2/expconv/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEBottleneck2/expbatch/gamma'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEBottleneck2/expbatch/beta'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&Bottleneck2/depthconv/depthwise_kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEBottleneck2/depthconv/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEBottleneck2/depthbatch/gamma'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEBottleneck2/depthbatch/beta'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEBottleneck2/projconv/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEBottleneck2/projconv/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEBottleneck2/projbatch/gamma'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEBottleneck2/projbatch/beta'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEBottleneck2/shortbatch/gamma'variables/40/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEBottleneck2/shortbatch/beta'variables/41/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE Bottleneck2/expbatch/moving_mean'variables/42/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$Bottleneck2/expbatch/moving_variance'variables/43/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"Bottleneck2/depthbatch/moving_mean'variables/44/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&Bottleneck2/depthbatch/moving_variance'variables/45/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!Bottleneck2/projbatch/moving_mean'variables/46/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%Bottleneck2/projbatch/moving_variance'variables/47/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"Bottleneck2/shortbatch/moving_mean'variables/48/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&Bottleneck2/shortbatch/moving_variance'variables/49/.ATTRIBUTES/VARIABLE_VALUE*

30
41
2
3
4
5
6
7
8
9
Ў10
Џ11
А12
Б13
В14
Г15
Д16
Е17*
z
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15*

0
1
2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

30
41*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
D
0
1
2
3
4
5
6
7*
5
N0
O1
P2
Q3
R4
S5
T6*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
џ	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
$
0
1
2
3*

0
1*
* 

 non_trainable_variables
Ёlayers
Ђmetrics
 Ѓlayer_regularization_losses
Єlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Ѕtrace_0
Іtrace_1* 

Їtrace_0
Јtrace_1* 
* 

0
1*

0
1*
* 

Љnon_trainable_variables
Њlayers
Ћmetrics
 Ќlayer_regularization_losses
­layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
$
0
1
2
3*

0
1*
* 

Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Гtrace_0
Дtrace_1* 

Еtrace_0
Жtrace_1* 
* 

0
1*

0
1*
* 

Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*
* 
* 
* 
$
0
1
2
3*

0
1*
* 

Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
Ђ	variables
Ѓtrainable_variables
Єregularization_losses
І__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses*

Сtrace_0
Тtrace_1* 

Уtrace_0
Фtrace_1* 
* 
$
0
1
2
3*

0
1*
* 

Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
Љ	variables
Њtrainable_variables
Ћregularization_losses
­__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses*

Ъtrace_0
Ыtrace_1* 

Ьtrace_0
Эtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
D
Ў0
Џ1
А2
Б3
В4
Г5
Д6
Е7*
5
h0
i1
j2
k3
l4
m5
n6*
* 
* 
* 
* 
* 
* 
* 

 0
Ё1*

 0
Ё1*
* 

Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses*
* 
* 
* 
$
Ђ0
Ѓ1
Ў2
Џ3*

Ђ0
Ѓ1*
* 

гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses*

иtrace_0
йtrace_1* 

кtrace_0
лtrace_1* 
* 

Є0
Ѕ1*

Є0
Ѕ1*
* 

мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
з	variables
иtrainable_variables
йregularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses*
* 
* 
* 
$
І0
Ї1
А2
Б3*

І0
Ї1*
* 

сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
о	variables
пtrainable_variables
рregularization_losses
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses*

цtrace_0
чtrace_1* 

шtrace_0
щtrace_1* 
* 

Ј0
Љ1*

Ј0
Љ1*
* 

ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
х	variables
цtrainable_variables
чregularization_losses
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses*
* 
* 
* 
$
Њ0
Ћ1
В2
Г3*

Њ0
Ћ1*
* 

яnon_trainable_variables
№layers
ёmetrics
 ђlayer_regularization_losses
ѓlayer_metrics
ь	variables
эtrainable_variables
юregularization_losses
№__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses*

єtrace_0
ѕtrace_1* 

іtrace_0
їtrace_1* 
* 
$
Ќ0
­1
Д2
Е3*

Ќ0
­1*
* 

јnon_trainable_variables
љlayers
њmetrics
 ћlayer_regularization_losses
ќlayer_metrics
ѓ	variables
єtrainable_variables
ѕregularization_losses
ї__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses*

§trace_0
ўtrace_1* 

џtrace_0
trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count

_fn_kwargs*
M
	variables
	keras_api

total

count

_fn_kwargs*
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Ў0
Џ1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

А0
Б1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

В0
Г1*
* 
* 
* 
* 
* 
* 
* 
* 

Д0
Е1*
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
}
VARIABLE_VALUEAdam/conv2d_18/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_18/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_34/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_34/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/Bottleneck1/expconv/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/Bottleneck1/expconv/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/Bottleneck1/expbatch/gamma/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/Bottleneck1/expbatch/beta/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/Bottleneck1/depthconv/depthwise_kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/Bottleneck1/depthconv/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/Bottleneck1/depthbatch/gamma/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/Bottleneck1/depthbatch/beta/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/Bottleneck1/projconv/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/Bottleneck1/projconv/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/Bottleneck1/projbatch/gamma/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/Bottleneck1/projbatch/beta/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/Bottleneck1/shortbatch/gamma/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/Bottleneck1/shortbatch/beta/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/Bottleneck2/expconv/kernel/mCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Bottleneck2/expconv/bias/mCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/Bottleneck2/expbatch/gamma/mCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/Bottleneck2/expbatch/beta/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/Bottleneck2/depthconv/depthwise_kernel/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/Bottleneck2/depthconv/bias/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/Bottleneck2/depthbatch/gamma/mCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/Bottleneck2/depthbatch/beta/mCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/Bottleneck2/projconv/kernel/mCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/Bottleneck2/projconv/bias/mCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/Bottleneck2/projbatch/gamma/mCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/Bottleneck2/projbatch/beta/mCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/Bottleneck2/shortbatch/gamma/mCvariables/40/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/Bottleneck2/shortbatch/beta/mCvariables/41/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_18/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_18/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_34/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_34/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/Bottleneck1/expconv/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/Bottleneck1/expconv/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/Bottleneck1/expbatch/gamma/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/Bottleneck1/expbatch/beta/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/Bottleneck1/depthconv/depthwise_kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/Bottleneck1/depthconv/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/Bottleneck1/depthbatch/gamma/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/Bottleneck1/depthbatch/beta/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/Bottleneck1/projconv/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/Bottleneck1/projconv/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/Bottleneck1/projbatch/gamma/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/Bottleneck1/projbatch/beta/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/Bottleneck1/shortbatch/gamma/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/Bottleneck1/shortbatch/beta/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/Bottleneck2/expconv/kernel/vCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Bottleneck2/expconv/bias/vCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/Bottleneck2/expbatch/gamma/vCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/Bottleneck2/expbatch/beta/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/Bottleneck2/depthconv/depthwise_kernel/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/Bottleneck2/depthconv/bias/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/Bottleneck2/depthbatch/gamma/vCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/Bottleneck2/depthbatch/beta/vCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/Bottleneck2/projconv/kernel/vCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/Bottleneck2/projconv/bias/vCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/Bottleneck2/projbatch/gamma/vCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/Bottleneck2/projbatch/beta/vCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE#Adam/Bottleneck2/shortbatch/gamma/vCvariables/40/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/Bottleneck2/shortbatch/beta/vCvariables/41/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ј7
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOp0batch_normalization_34/gamma/Read/ReadVariableOp/batch_normalization_34/beta/Read/ReadVariableOp6batch_normalization_34/moving_mean/Read/ReadVariableOp:batch_normalization_34/moving_variance/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp.Bottleneck1/expconv/kernel/Read/ReadVariableOp,Bottleneck1/expconv/bias/Read/ReadVariableOp.Bottleneck1/expbatch/gamma/Read/ReadVariableOp-Bottleneck1/expbatch/beta/Read/ReadVariableOp:Bottleneck1/depthconv/depthwise_kernel/Read/ReadVariableOp.Bottleneck1/depthconv/bias/Read/ReadVariableOp0Bottleneck1/depthbatch/gamma/Read/ReadVariableOp/Bottleneck1/depthbatch/beta/Read/ReadVariableOp/Bottleneck1/projconv/kernel/Read/ReadVariableOp-Bottleneck1/projconv/bias/Read/ReadVariableOp/Bottleneck1/projbatch/gamma/Read/ReadVariableOp.Bottleneck1/projbatch/beta/Read/ReadVariableOp0Bottleneck1/shortbatch/gamma/Read/ReadVariableOp/Bottleneck1/shortbatch/beta/Read/ReadVariableOp4Bottleneck1/expbatch/moving_mean/Read/ReadVariableOp8Bottleneck1/expbatch/moving_variance/Read/ReadVariableOp6Bottleneck1/depthbatch/moving_mean/Read/ReadVariableOp:Bottleneck1/depthbatch/moving_variance/Read/ReadVariableOp5Bottleneck1/projbatch/moving_mean/Read/ReadVariableOp9Bottleneck1/projbatch/moving_variance/Read/ReadVariableOp6Bottleneck1/shortbatch/moving_mean/Read/ReadVariableOp:Bottleneck1/shortbatch/moving_variance/Read/ReadVariableOp.Bottleneck2/expconv/kernel/Read/ReadVariableOp,Bottleneck2/expconv/bias/Read/ReadVariableOp.Bottleneck2/expbatch/gamma/Read/ReadVariableOp-Bottleneck2/expbatch/beta/Read/ReadVariableOp:Bottleneck2/depthconv/depthwise_kernel/Read/ReadVariableOp.Bottleneck2/depthconv/bias/Read/ReadVariableOp0Bottleneck2/depthbatch/gamma/Read/ReadVariableOp/Bottleneck2/depthbatch/beta/Read/ReadVariableOp/Bottleneck2/projconv/kernel/Read/ReadVariableOp-Bottleneck2/projconv/bias/Read/ReadVariableOp/Bottleneck2/projbatch/gamma/Read/ReadVariableOp.Bottleneck2/projbatch/beta/Read/ReadVariableOp0Bottleneck2/shortbatch/gamma/Read/ReadVariableOp/Bottleneck2/shortbatch/beta/Read/ReadVariableOp4Bottleneck2/expbatch/moving_mean/Read/ReadVariableOp8Bottleneck2/expbatch/moving_variance/Read/ReadVariableOp6Bottleneck2/depthbatch/moving_mean/Read/ReadVariableOp:Bottleneck2/depthbatch/moving_variance/Read/ReadVariableOp5Bottleneck2/projbatch/moving_mean/Read/ReadVariableOp9Bottleneck2/projbatch/moving_variance/Read/ReadVariableOp6Bottleneck2/shortbatch/moving_mean/Read/ReadVariableOp:Bottleneck2/shortbatch/moving_variance/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_18/kernel/m/Read/ReadVariableOp)Adam/conv2d_18/bias/m/Read/ReadVariableOp7Adam/batch_normalization_34/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_34/beta/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp5Adam/Bottleneck1/expconv/kernel/m/Read/ReadVariableOp3Adam/Bottleneck1/expconv/bias/m/Read/ReadVariableOp5Adam/Bottleneck1/expbatch/gamma/m/Read/ReadVariableOp4Adam/Bottleneck1/expbatch/beta/m/Read/ReadVariableOpAAdam/Bottleneck1/depthconv/depthwise_kernel/m/Read/ReadVariableOp5Adam/Bottleneck1/depthconv/bias/m/Read/ReadVariableOp7Adam/Bottleneck1/depthbatch/gamma/m/Read/ReadVariableOp6Adam/Bottleneck1/depthbatch/beta/m/Read/ReadVariableOp6Adam/Bottleneck1/projconv/kernel/m/Read/ReadVariableOp4Adam/Bottleneck1/projconv/bias/m/Read/ReadVariableOp6Adam/Bottleneck1/projbatch/gamma/m/Read/ReadVariableOp5Adam/Bottleneck1/projbatch/beta/m/Read/ReadVariableOp7Adam/Bottleneck1/shortbatch/gamma/m/Read/ReadVariableOp6Adam/Bottleneck1/shortbatch/beta/m/Read/ReadVariableOp5Adam/Bottleneck2/expconv/kernel/m/Read/ReadVariableOp3Adam/Bottleneck2/expconv/bias/m/Read/ReadVariableOp5Adam/Bottleneck2/expbatch/gamma/m/Read/ReadVariableOp4Adam/Bottleneck2/expbatch/beta/m/Read/ReadVariableOpAAdam/Bottleneck2/depthconv/depthwise_kernel/m/Read/ReadVariableOp5Adam/Bottleneck2/depthconv/bias/m/Read/ReadVariableOp7Adam/Bottleneck2/depthbatch/gamma/m/Read/ReadVariableOp6Adam/Bottleneck2/depthbatch/beta/m/Read/ReadVariableOp6Adam/Bottleneck2/projconv/kernel/m/Read/ReadVariableOp4Adam/Bottleneck2/projconv/bias/m/Read/ReadVariableOp6Adam/Bottleneck2/projbatch/gamma/m/Read/ReadVariableOp5Adam/Bottleneck2/projbatch/beta/m/Read/ReadVariableOp7Adam/Bottleneck2/shortbatch/gamma/m/Read/ReadVariableOp6Adam/Bottleneck2/shortbatch/beta/m/Read/ReadVariableOp+Adam/conv2d_18/kernel/v/Read/ReadVariableOp)Adam/conv2d_18/bias/v/Read/ReadVariableOp7Adam/batch_normalization_34/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_34/beta/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp5Adam/Bottleneck1/expconv/kernel/v/Read/ReadVariableOp3Adam/Bottleneck1/expconv/bias/v/Read/ReadVariableOp5Adam/Bottleneck1/expbatch/gamma/v/Read/ReadVariableOp4Adam/Bottleneck1/expbatch/beta/v/Read/ReadVariableOpAAdam/Bottleneck1/depthconv/depthwise_kernel/v/Read/ReadVariableOp5Adam/Bottleneck1/depthconv/bias/v/Read/ReadVariableOp7Adam/Bottleneck1/depthbatch/gamma/v/Read/ReadVariableOp6Adam/Bottleneck1/depthbatch/beta/v/Read/ReadVariableOp6Adam/Bottleneck1/projconv/kernel/v/Read/ReadVariableOp4Adam/Bottleneck1/projconv/bias/v/Read/ReadVariableOp6Adam/Bottleneck1/projbatch/gamma/v/Read/ReadVariableOp5Adam/Bottleneck1/projbatch/beta/v/Read/ReadVariableOp7Adam/Bottleneck1/shortbatch/gamma/v/Read/ReadVariableOp6Adam/Bottleneck1/shortbatch/beta/v/Read/ReadVariableOp5Adam/Bottleneck2/expconv/kernel/v/Read/ReadVariableOp3Adam/Bottleneck2/expconv/bias/v/Read/ReadVariableOp5Adam/Bottleneck2/expbatch/gamma/v/Read/ReadVariableOp4Adam/Bottleneck2/expbatch/beta/v/Read/ReadVariableOpAAdam/Bottleneck2/depthconv/depthwise_kernel/v/Read/ReadVariableOp5Adam/Bottleneck2/depthconv/bias/v/Read/ReadVariableOp7Adam/Bottleneck2/depthbatch/gamma/v/Read/ReadVariableOp6Adam/Bottleneck2/depthbatch/beta/v/Read/ReadVariableOp6Adam/Bottleneck2/projconv/kernel/v/Read/ReadVariableOp4Adam/Bottleneck2/projconv/bias/v/Read/ReadVariableOp6Adam/Bottleneck2/projbatch/gamma/v/Read/ReadVariableOp5Adam/Bottleneck2/projbatch/beta/v/Read/ReadVariableOp7Adam/Bottleneck2/shortbatch/gamma/v/Read/ReadVariableOp6Adam/Bottleneck2/shortbatch/beta/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_51882
ћ"
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_18/kernelconv2d_18/biasbatch_normalization_34/gammabatch_normalization_34/beta"batch_normalization_34/moving_mean&batch_normalization_34/moving_variancedense_2/kerneldense_2/biasBottleneck1/expconv/kernelBottleneck1/expconv/biasBottleneck1/expbatch/gammaBottleneck1/expbatch/beta&Bottleneck1/depthconv/depthwise_kernelBottleneck1/depthconv/biasBottleneck1/depthbatch/gammaBottleneck1/depthbatch/betaBottleneck1/projconv/kernelBottleneck1/projconv/biasBottleneck1/projbatch/gammaBottleneck1/projbatch/betaBottleneck1/shortbatch/gammaBottleneck1/shortbatch/beta Bottleneck1/expbatch/moving_mean$Bottleneck1/expbatch/moving_variance"Bottleneck1/depthbatch/moving_mean&Bottleneck1/depthbatch/moving_variance!Bottleneck1/projbatch/moving_mean%Bottleneck1/projbatch/moving_variance"Bottleneck1/shortbatch/moving_mean&Bottleneck1/shortbatch/moving_varianceBottleneck2/expconv/kernelBottleneck2/expconv/biasBottleneck2/expbatch/gammaBottleneck2/expbatch/beta&Bottleneck2/depthconv/depthwise_kernelBottleneck2/depthconv/biasBottleneck2/depthbatch/gammaBottleneck2/depthbatch/betaBottleneck2/projconv/kernelBottleneck2/projconv/biasBottleneck2/projbatch/gammaBottleneck2/projbatch/betaBottleneck2/shortbatch/gammaBottleneck2/shortbatch/beta Bottleneck2/expbatch/moving_mean$Bottleneck2/expbatch/moving_variance"Bottleneck2/depthbatch/moving_mean&Bottleneck2/depthbatch/moving_variance!Bottleneck2/projbatch/moving_mean%Bottleneck2/projbatch/moving_variance"Bottleneck2/shortbatch/moving_mean&Bottleneck2/shortbatch/moving_variance	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotal_2count_2total_1count_1totalcountAdam/conv2d_18/kernel/mAdam/conv2d_18/bias/m#Adam/batch_normalization_34/gamma/m"Adam/batch_normalization_34/beta/mAdam/dense_2/kernel/mAdam/dense_2/bias/m!Adam/Bottleneck1/expconv/kernel/mAdam/Bottleneck1/expconv/bias/m!Adam/Bottleneck1/expbatch/gamma/m Adam/Bottleneck1/expbatch/beta/m-Adam/Bottleneck1/depthconv/depthwise_kernel/m!Adam/Bottleneck1/depthconv/bias/m#Adam/Bottleneck1/depthbatch/gamma/m"Adam/Bottleneck1/depthbatch/beta/m"Adam/Bottleneck1/projconv/kernel/m Adam/Bottleneck1/projconv/bias/m"Adam/Bottleneck1/projbatch/gamma/m!Adam/Bottleneck1/projbatch/beta/m#Adam/Bottleneck1/shortbatch/gamma/m"Adam/Bottleneck1/shortbatch/beta/m!Adam/Bottleneck2/expconv/kernel/mAdam/Bottleneck2/expconv/bias/m!Adam/Bottleneck2/expbatch/gamma/m Adam/Bottleneck2/expbatch/beta/m-Adam/Bottleneck2/depthconv/depthwise_kernel/m!Adam/Bottleneck2/depthconv/bias/m#Adam/Bottleneck2/depthbatch/gamma/m"Adam/Bottleneck2/depthbatch/beta/m"Adam/Bottleneck2/projconv/kernel/m Adam/Bottleneck2/projconv/bias/m"Adam/Bottleneck2/projbatch/gamma/m!Adam/Bottleneck2/projbatch/beta/m#Adam/Bottleneck2/shortbatch/gamma/m"Adam/Bottleneck2/shortbatch/beta/mAdam/conv2d_18/kernel/vAdam/conv2d_18/bias/v#Adam/batch_normalization_34/gamma/v"Adam/batch_normalization_34/beta/vAdam/dense_2/kernel/vAdam/dense_2/bias/v!Adam/Bottleneck1/expconv/kernel/vAdam/Bottleneck1/expconv/bias/v!Adam/Bottleneck1/expbatch/gamma/v Adam/Bottleneck1/expbatch/beta/v-Adam/Bottleneck1/depthconv/depthwise_kernel/v!Adam/Bottleneck1/depthconv/bias/v#Adam/Bottleneck1/depthbatch/gamma/v"Adam/Bottleneck1/depthbatch/beta/v"Adam/Bottleneck1/projconv/kernel/v Adam/Bottleneck1/projconv/bias/v"Adam/Bottleneck1/projbatch/gamma/v!Adam/Bottleneck1/projbatch/beta/v#Adam/Bottleneck1/shortbatch/gamma/v"Adam/Bottleneck1/shortbatch/beta/v!Adam/Bottleneck2/expconv/kernel/vAdam/Bottleneck2/expconv/bias/v!Adam/Bottleneck2/expbatch/gamma/v Adam/Bottleneck2/expbatch/beta/v-Adam/Bottleneck2/depthconv/depthwise_kernel/v!Adam/Bottleneck2/depthconv/bias/v#Adam/Bottleneck2/depthbatch/gamma/v"Adam/Bottleneck2/depthbatch/beta/v"Adam/Bottleneck2/projconv/kernel/v Adam/Bottleneck2/projconv/bias/v"Adam/Bottleneck2/projbatch/gamma/v!Adam/Bottleneck2/projbatch/beta/v#Adam/Bottleneck2/shortbatch/gamma/v"Adam/Bottleneck2/shortbatch/beta/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_52282Э
ї
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_50930

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџc

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П

D__inference_projbatch_layer_call_and_return_conditional_losses_51141

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Во
 6
B__inference_model_2_layer_call_and_return_conditional_losses_50195

inputsB
(conv2d_18_conv2d_readvariableop_resource:7
)conv2d_18_biasadd_readvariableop_resource:<
.batch_normalization_34_readvariableop_resource:>
0batch_normalization_34_readvariableop_1_resource:M
?batch_normalization_34_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:L
2bottleneck1_expconv_conv2d_readvariableop_resource:A
3bottleneck1_expconv_biasadd_readvariableop_resource::
,bottleneck1_expbatch_readvariableop_resource:<
.bottleneck1_expbatch_readvariableop_1_resource:K
=bottleneck1_expbatch_fusedbatchnormv3_readvariableop_resource:M
?bottleneck1_expbatch_fusedbatchnormv3_readvariableop_1_resource:Q
7bottleneck1_depthconv_depthwise_readvariableop_resource:C
5bottleneck1_depthconv_biasadd_readvariableop_resource:<
.bottleneck1_depthbatch_readvariableop_resource:>
0bottleneck1_depthbatch_readvariableop_1_resource:M
?bottleneck1_depthbatch_fusedbatchnormv3_readvariableop_resource:O
Abottleneck1_depthbatch_fusedbatchnormv3_readvariableop_1_resource:M
3bottleneck1_projconv_conv2d_readvariableop_resource:B
4bottleneck1_projconv_biasadd_readvariableop_resource:;
-bottleneck1_projbatch_readvariableop_resource:=
/bottleneck1_projbatch_readvariableop_1_resource:L
>bottleneck1_projbatch_fusedbatchnormv3_readvariableop_resource:N
@bottleneck1_projbatch_fusedbatchnormv3_readvariableop_1_resource:<
.bottleneck1_shortbatch_readvariableop_resource:>
0bottleneck1_shortbatch_readvariableop_1_resource:M
?bottleneck1_shortbatch_fusedbatchnormv3_readvariableop_resource:O
Abottleneck1_shortbatch_fusedbatchnormv3_readvariableop_1_resource:L
2bottleneck2_expconv_conv2d_readvariableop_resource:0A
3bottleneck2_expconv_biasadd_readvariableop_resource:0:
,bottleneck2_expbatch_readvariableop_resource:0<
.bottleneck2_expbatch_readvariableop_1_resource:0K
=bottleneck2_expbatch_fusedbatchnormv3_readvariableop_resource:0M
?bottleneck2_expbatch_fusedbatchnormv3_readvariableop_1_resource:0Q
7bottleneck2_depthconv_depthwise_readvariableop_resource:0C
5bottleneck2_depthconv_biasadd_readvariableop_resource:0<
.bottleneck2_depthbatch_readvariableop_resource:0>
0bottleneck2_depthbatch_readvariableop_1_resource:0M
?bottleneck2_depthbatch_fusedbatchnormv3_readvariableop_resource:0O
Abottleneck2_depthbatch_fusedbatchnormv3_readvariableop_1_resource:0M
3bottleneck2_projconv_conv2d_readvariableop_resource:0B
4bottleneck2_projconv_biasadd_readvariableop_resource:;
-bottleneck2_projbatch_readvariableop_resource:=
/bottleneck2_projbatch_readvariableop_1_resource:L
>bottleneck2_projbatch_fusedbatchnormv3_readvariableop_resource:N
@bottleneck2_projbatch_fusedbatchnormv3_readvariableop_1_resource:<
.bottleneck2_shortbatch_readvariableop_resource:>
0bottleneck2_shortbatch_readvariableop_1_resource:M
?bottleneck2_shortbatch_fusedbatchnormv3_readvariableop_resource:O
Abottleneck2_shortbatch_fusedbatchnormv3_readvariableop_1_resource:9
&dense_2_matmul_readvariableop_resource:		5
'dense_2_biasadd_readvariableop_resource:
identityЂ%Bottleneck1/depthbatch/AssignNewValueЂ'Bottleneck1/depthbatch/AssignNewValue_1Ђ6Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOpЂ8Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_1Ђ%Bottleneck1/depthbatch/ReadVariableOpЂ'Bottleneck1/depthbatch/ReadVariableOp_1Ђ,Bottleneck1/depthconv/BiasAdd/ReadVariableOpЂ.Bottleneck1/depthconv/depthwise/ReadVariableOpЂ#Bottleneck1/expbatch/AssignNewValueЂ%Bottleneck1/expbatch/AssignNewValue_1Ђ4Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOpЂ6Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_1Ђ#Bottleneck1/expbatch/ReadVariableOpЂ%Bottleneck1/expbatch/ReadVariableOp_1Ђ*Bottleneck1/expconv/BiasAdd/ReadVariableOpЂ)Bottleneck1/expconv/Conv2D/ReadVariableOpЂ$Bottleneck1/projbatch/AssignNewValueЂ&Bottleneck1/projbatch/AssignNewValue_1Ђ5Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOpЂ7Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_1Ђ$Bottleneck1/projbatch/ReadVariableOpЂ&Bottleneck1/projbatch/ReadVariableOp_1Ђ+Bottleneck1/projconv/BiasAdd/ReadVariableOpЂ*Bottleneck1/projconv/Conv2D/ReadVariableOpЂ%Bottleneck1/shortbatch/AssignNewValueЂ'Bottleneck1/shortbatch/AssignNewValue_1Ђ6Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOpЂ8Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_1Ђ%Bottleneck1/shortbatch/ReadVariableOpЂ'Bottleneck1/shortbatch/ReadVariableOp_1Ђ%Bottleneck2/depthbatch/AssignNewValueЂ'Bottleneck2/depthbatch/AssignNewValue_1Ђ6Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOpЂ8Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_1Ђ%Bottleneck2/depthbatch/ReadVariableOpЂ'Bottleneck2/depthbatch/ReadVariableOp_1Ђ,Bottleneck2/depthconv/BiasAdd/ReadVariableOpЂ.Bottleneck2/depthconv/depthwise/ReadVariableOpЂ#Bottleneck2/expbatch/AssignNewValueЂ%Bottleneck2/expbatch/AssignNewValue_1Ђ4Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOpЂ6Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_1Ђ#Bottleneck2/expbatch/ReadVariableOpЂ%Bottleneck2/expbatch/ReadVariableOp_1Ђ*Bottleneck2/expconv/BiasAdd/ReadVariableOpЂ)Bottleneck2/expconv/Conv2D/ReadVariableOpЂ$Bottleneck2/projbatch/AssignNewValueЂ&Bottleneck2/projbatch/AssignNewValue_1Ђ5Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOpЂ7Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_1Ђ$Bottleneck2/projbatch/ReadVariableOpЂ&Bottleneck2/projbatch/ReadVariableOp_1Ђ+Bottleneck2/projconv/BiasAdd/ReadVariableOpЂ*Bottleneck2/projconv/Conv2D/ReadVariableOpЂ%Bottleneck2/shortbatch/AssignNewValueЂ'Bottleneck2/shortbatch/AssignNewValue_1Ђ6Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOpЂ8Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_1Ђ%Bottleneck2/shortbatch/ReadVariableOpЂ'Bottleneck2/shortbatch/ReadVariableOp_1Ђ%batch_normalization_34/AssignNewValueЂ'batch_normalization_34/AssignNewValue_1Ђ6batch_normalization_34/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_34/ReadVariableOpЂ'batch_normalization_34/ReadVariableOp_1Ђ conv2d_18/BiasAdd/ReadVariableOpЂconv2d_18/Conv2D/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpf
tf.identity_2/IdentityIdentityinputs*
T0*1
_output_shapes
:џџџџџџџџџррW
rescaling_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;Y
rescaling_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
rescaling_2/mulMultf.identity_2/Identity:output:0rescaling_2/Cast/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџрр
rescaling_2/addAddV2rescaling_2/mul:z:0rescaling_2/Cast_1/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџрр
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0К
conv2d_18/Conv2DConv2Drescaling_2/add:z:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџpp*
paddingSAME*
strides

 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџpp
%batch_normalization_34/ReadVariableOpReadVariableOp.batch_normalization_34_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_34/ReadVariableOp_1ReadVariableOp0batch_normalization_34_readvariableop_1_resource*
_output_shapes
:*
dtype0В
6batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ж
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ы
'batch_normalization_34/FusedBatchNormV3FusedBatchNormV3conv2d_18/BiasAdd:output:0-batch_normalization_34/ReadVariableOp:value:0/batch_normalization_34/ReadVariableOp_1:value:0>batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџpp:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_34/AssignNewValueAssignVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource4batch_normalization_34/FusedBatchNormV3:batch_mean:07^batch_normalization_34/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_34/AssignNewValue_1AssignVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_34/FusedBatchNormV3:batch_variance:09^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
activation_2/ReluRelu+batch_normalization_34/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџppА
max_pooling2d_6/MaxPoolMaxPoolactivation_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ88*
ksize
*
paddingVALID*
strides
\
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?
dropout_6/dropout/MulMul max_pooling2d_6/MaxPool:output:0 dropout_6/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ88g
dropout_6/dropout/ShapeShape max_pooling2d_6/MaxPool:output:0*
T0*
_output_shapes
:Д
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
dtype0*

seed*e
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<Ь
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ88
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ88
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88Є
)Bottleneck1/expconv/Conv2D/ReadVariableOpReadVariableOp2bottleneck1_expconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ж
Bottleneck1/expconv/Conv2DConv2Ddropout_6/dropout/Mul_1:z:01Bottleneck1/expconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

*Bottleneck1/expconv/BiasAdd/ReadVariableOpReadVariableOp3bottleneck1_expconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Й
Bottleneck1/expconv/BiasAddBiasAdd#Bottleneck1/expconv/Conv2D:output:02Bottleneck1/expconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88
#Bottleneck1/expbatch/ReadVariableOpReadVariableOp,bottleneck1_expbatch_readvariableop_resource*
_output_shapes
:*
dtype0
%Bottleneck1/expbatch/ReadVariableOp_1ReadVariableOp.bottleneck1_expbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0Ў
4Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp=bottleneck1_expbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0В
6Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp?bottleneck1_expbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ы
%Bottleneck1/expbatch/FusedBatchNormV3FusedBatchNormV3$Bottleneck1/expconv/BiasAdd:output:0+Bottleneck1/expbatch/ReadVariableOp:value:0-Bottleneck1/expbatch/ReadVariableOp_1:value:0<Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp:value:0>Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
exponential_avg_factor%
з#<
#Bottleneck1/expbatch/AssignNewValueAssignVariableOp=bottleneck1_expbatch_fusedbatchnormv3_readvariableop_resource2Bottleneck1/expbatch/FusedBatchNormV3:batch_mean:05^Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Є
%Bottleneck1/expbatch/AssignNewValue_1AssignVariableOp?bottleneck1_expbatch_fusedbatchnormv3_readvariableop_1_resource6Bottleneck1/expbatch/FusedBatchNormV3:batch_variance:07^Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
Bottleneck1/activation/ReluRelu)Bottleneck1/expbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88Ў
.Bottleneck1/depthconv/depthwise/ReadVariableOpReadVariableOp7bottleneck1_depthconv_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0~
%Bottleneck1/depthconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            ~
-Bottleneck1/depthconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      §
Bottleneck1/depthconv/depthwiseDepthwiseConv2dNative)Bottleneck1/activation/Relu:activations:06Bottleneck1/depthconv/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

,Bottleneck1/depthconv/BiasAdd/ReadVariableOpReadVariableOp5bottleneck1_depthconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Т
Bottleneck1/depthconv/BiasAddBiasAdd(Bottleneck1/depthconv/depthwise:output:04Bottleneck1/depthconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88
%Bottleneck1/depthbatch/ReadVariableOpReadVariableOp.bottleneck1_depthbatch_readvariableop_resource*
_output_shapes
:*
dtype0
'Bottleneck1/depthbatch/ReadVariableOp_1ReadVariableOp0bottleneck1_depthbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0В
6Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp?bottleneck1_depthbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ж
8Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbottleneck1_depthbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0з
'Bottleneck1/depthbatch/FusedBatchNormV3FusedBatchNormV3&Bottleneck1/depthconv/BiasAdd:output:0-Bottleneck1/depthbatch/ReadVariableOp:value:0/Bottleneck1/depthbatch/ReadVariableOp_1:value:0>Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp:value:0@Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%Bottleneck1/depthbatch/AssignNewValueAssignVariableOp?bottleneck1_depthbatch_fusedbatchnormv3_readvariableop_resource4Bottleneck1/depthbatch/FusedBatchNormV3:batch_mean:07^Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'Bottleneck1/depthbatch/AssignNewValue_1AssignVariableOpAbottleneck1_depthbatch_fusedbatchnormv3_readvariableop_1_resource8Bottleneck1/depthbatch/FusedBatchNormV3:batch_variance:09^Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
Bottleneck1/activation_1/ReluRelu+Bottleneck1/depthbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88І
*Bottleneck1/projconv/Conv2D/ReadVariableOpReadVariableOp3bottleneck1_projconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ш
Bottleneck1/projconv/Conv2DConv2D+Bottleneck1/activation_1/Relu:activations:02Bottleneck1/projconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

+Bottleneck1/projconv/BiasAdd/ReadVariableOpReadVariableOp4bottleneck1_projconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0М
Bottleneck1/projconv/BiasAddBiasAdd$Bottleneck1/projconv/Conv2D:output:03Bottleneck1/projconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88
$Bottleneck1/projbatch/ReadVariableOpReadVariableOp-bottleneck1_projbatch_readvariableop_resource*
_output_shapes
:*
dtype0
&Bottleneck1/projbatch/ReadVariableOp_1ReadVariableOp/bottleneck1_projbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0А
5Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp>bottleneck1_projbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Д
7Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@bottleneck1_projbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0б
&Bottleneck1/projbatch/FusedBatchNormV3FusedBatchNormV3%Bottleneck1/projconv/BiasAdd:output:0,Bottleneck1/projbatch/ReadVariableOp:value:0.Bottleneck1/projbatch/ReadVariableOp_1:value:0=Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp:value:0?Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
exponential_avg_factor%
з#<
$Bottleneck1/projbatch/AssignNewValueAssignVariableOp>bottleneck1_projbatch_fusedbatchnormv3_readvariableop_resource3Bottleneck1/projbatch/FusedBatchNormV3:batch_mean:06^Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ј
&Bottleneck1/projbatch/AssignNewValue_1AssignVariableOp@bottleneck1_projbatch_fusedbatchnormv3_readvariableop_1_resource7Bottleneck1/projbatch/FusedBatchNormV3:batch_variance:08^Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
%Bottleneck1/shortbatch/ReadVariableOpReadVariableOp.bottleneck1_shortbatch_readvariableop_resource*
_output_shapes
:*
dtype0
'Bottleneck1/shortbatch/ReadVariableOp_1ReadVariableOp0bottleneck1_shortbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0В
6Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp?bottleneck1_shortbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ж
8Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbottleneck1_shortbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ь
'Bottleneck1/shortbatch/FusedBatchNormV3FusedBatchNormV3dropout_6/dropout/Mul_1:z:0-Bottleneck1/shortbatch/ReadVariableOp:value:0/Bottleneck1/shortbatch/ReadVariableOp_1:value:0>Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp:value:0@Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%Bottleneck1/shortbatch/AssignNewValueAssignVariableOp?bottleneck1_shortbatch_fusedbatchnormv3_readvariableop_resource4Bottleneck1/shortbatch/FusedBatchNormV3:batch_mean:07^Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'Bottleneck1/shortbatch/AssignNewValue_1AssignVariableOpAbottleneck1_shortbatch_fusedbatchnormv3_readvariableop_1_resource8Bottleneck1/shortbatch/FusedBatchNormV3:batch_variance:09^Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Џ
Bottleneck1/add/addAddV2*Bottleneck1/projbatch/FusedBatchNormV3:y:0+Bottleneck1/shortbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88x
Bottleneck1/activation_2/ReluReluBottleneck1/add/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ88М
max_pooling2d_7/MaxPoolMaxPool+Bottleneck1/activation_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
\
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?
dropout_7/dropout/MulMul max_pooling2d_7/MaxPool:output:0 dropout_7/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџg
dropout_7/dropout/ShapeShape max_pooling2d_7/MaxPool:output:0*
T0*
_output_shapes
:С
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*

seed**
seed2e
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<Ь
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџЄ
)Bottleneck2/expconv/Conv2D/ReadVariableOpReadVariableOp2bottleneck2_expconv_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0ж
Bottleneck2/expconv/Conv2DConv2Ddropout_7/dropout/Mul_1:z:01Bottleneck2/expconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides

*Bottleneck2/expconv/BiasAdd/ReadVariableOpReadVariableOp3bottleneck2_expconv_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0Й
Bottleneck2/expconv/BiasAddBiasAdd#Bottleneck2/expconv/Conv2D:output:02Bottleneck2/expconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0
#Bottleneck2/expbatch/ReadVariableOpReadVariableOp,bottleneck2_expbatch_readvariableop_resource*
_output_shapes
:0*
dtype0
%Bottleneck2/expbatch/ReadVariableOp_1ReadVariableOp.bottleneck2_expbatch_readvariableop_1_resource*
_output_shapes
:0*
dtype0Ў
4Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp=bottleneck2_expbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0В
6Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp?bottleneck2_expbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0Ы
%Bottleneck2/expbatch/FusedBatchNormV3FusedBatchNormV3$Bottleneck2/expconv/BiasAdd:output:0+Bottleneck2/expbatch/ReadVariableOp:value:0-Bottleneck2/expbatch/ReadVariableOp_1:value:0<Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp:value:0>Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<
#Bottleneck2/expbatch/AssignNewValueAssignVariableOp=bottleneck2_expbatch_fusedbatchnormv3_readvariableop_resource2Bottleneck2/expbatch/FusedBatchNormV3:batch_mean:05^Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Є
%Bottleneck2/expbatch/AssignNewValue_1AssignVariableOp?bottleneck2_expbatch_fusedbatchnormv3_readvariableop_1_resource6Bottleneck2/expbatch/FusedBatchNormV3:batch_variance:07^Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
Bottleneck2/activation_3/ReluRelu)Bottleneck2/expbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ0Ў
.Bottleneck2/depthconv/depthwise/ReadVariableOpReadVariableOp7bottleneck2_depthconv_depthwise_readvariableop_resource*&
_output_shapes
:0*
dtype0~
%Bottleneck2/depthconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      ~
-Bottleneck2/depthconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      џ
Bottleneck2/depthconv/depthwiseDepthwiseConv2dNative+Bottleneck2/activation_3/Relu:activations:06Bottleneck2/depthconv/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides

,Bottleneck2/depthconv/BiasAdd/ReadVariableOpReadVariableOp5bottleneck2_depthconv_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0Т
Bottleneck2/depthconv/BiasAddBiasAdd(Bottleneck2/depthconv/depthwise:output:04Bottleneck2/depthconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0
%Bottleneck2/depthbatch/ReadVariableOpReadVariableOp.bottleneck2_depthbatch_readvariableop_resource*
_output_shapes
:0*
dtype0
'Bottleneck2/depthbatch/ReadVariableOp_1ReadVariableOp0bottleneck2_depthbatch_readvariableop_1_resource*
_output_shapes
:0*
dtype0В
6Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp?bottleneck2_depthbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0Ж
8Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbottleneck2_depthbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0з
'Bottleneck2/depthbatch/FusedBatchNormV3FusedBatchNormV3&Bottleneck2/depthconv/BiasAdd:output:0-Bottleneck2/depthbatch/ReadVariableOp:value:0/Bottleneck2/depthbatch/ReadVariableOp_1:value:0>Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp:value:0@Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%Bottleneck2/depthbatch/AssignNewValueAssignVariableOp?bottleneck2_depthbatch_fusedbatchnormv3_readvariableop_resource4Bottleneck2/depthbatch/FusedBatchNormV3:batch_mean:07^Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'Bottleneck2/depthbatch/AssignNewValue_1AssignVariableOpAbottleneck2_depthbatch_fusedbatchnormv3_readvariableop_1_resource8Bottleneck2/depthbatch/FusedBatchNormV3:batch_variance:09^Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
Bottleneck2/activation_4/ReluRelu+Bottleneck2/depthbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ0І
*Bottleneck2/projconv/Conv2D/ReadVariableOpReadVariableOp3bottleneck2_projconv_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0ш
Bottleneck2/projconv/Conv2DConv2D+Bottleneck2/activation_4/Relu:activations:02Bottleneck2/projconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

+Bottleneck2/projconv/BiasAdd/ReadVariableOpReadVariableOp4bottleneck2_projconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0М
Bottleneck2/projconv/BiasAddBiasAdd$Bottleneck2/projconv/Conv2D:output:03Bottleneck2/projconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
$Bottleneck2/projbatch/ReadVariableOpReadVariableOp-bottleneck2_projbatch_readvariableop_resource*
_output_shapes
:*
dtype0
&Bottleneck2/projbatch/ReadVariableOp_1ReadVariableOp/bottleneck2_projbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0А
5Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp>bottleneck2_projbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Д
7Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@bottleneck2_projbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0б
&Bottleneck2/projbatch/FusedBatchNormV3FusedBatchNormV3%Bottleneck2/projconv/BiasAdd:output:0,Bottleneck2/projbatch/ReadVariableOp:value:0.Bottleneck2/projbatch/ReadVariableOp_1:value:0=Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp:value:0?Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<
$Bottleneck2/projbatch/AssignNewValueAssignVariableOp>bottleneck2_projbatch_fusedbatchnormv3_readvariableop_resource3Bottleneck2/projbatch/FusedBatchNormV3:batch_mean:06^Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ј
&Bottleneck2/projbatch/AssignNewValue_1AssignVariableOp@bottleneck2_projbatch_fusedbatchnormv3_readvariableop_1_resource7Bottleneck2/projbatch/FusedBatchNormV3:batch_variance:08^Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
%Bottleneck2/shortbatch/ReadVariableOpReadVariableOp.bottleneck2_shortbatch_readvariableop_resource*
_output_shapes
:*
dtype0
'Bottleneck2/shortbatch/ReadVariableOp_1ReadVariableOp0bottleneck2_shortbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0В
6Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp?bottleneck2_shortbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ж
8Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbottleneck2_shortbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ь
'Bottleneck2/shortbatch/FusedBatchNormV3FusedBatchNormV3dropout_7/dropout/Mul_1:z:0-Bottleneck2/shortbatch/ReadVariableOp:value:0/Bottleneck2/shortbatch/ReadVariableOp_1:value:0>Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp:value:0@Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%Bottleneck2/shortbatch/AssignNewValueAssignVariableOp?bottleneck2_shortbatch_fusedbatchnormv3_readvariableop_resource4Bottleneck2/shortbatch/FusedBatchNormV3:batch_mean:07^Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'Bottleneck2/shortbatch/AssignNewValue_1AssignVariableOpAbottleneck2_shortbatch_fusedbatchnormv3_readvariableop_1_resource8Bottleneck2/shortbatch/FusedBatchNormV3:batch_variance:09^Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Б
Bottleneck2/add_1/addAddV2*Bottleneck2/projbatch/FusedBatchNormV3:y:0+Bottleneck2/shortbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџz
Bottleneck2/activation_5/ReluReluBottleneck2/add_1/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџМ
max_pooling2d_8/MaxPoolMaxPool+Bottleneck2/activation_5/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
\
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?
dropout_8/dropout/MulMul max_pooling2d_8/MaxPool:output:0 dropout_8/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџg
dropout_8/dropout/ShapeShape max_pooling2d_8/MaxPool:output:0*
T0*
_output_shapes
:С
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*

seed**
seed2e
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=Ь
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  
flatten_2/ReshapeReshapedropout_8/dropout/Mul_1:z:0flatten_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:		*
dtype0
dense_2/MatMulMatMulflatten_2/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџf
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЈ
NoOpNoOp&^Bottleneck1/depthbatch/AssignNewValue(^Bottleneck1/depthbatch/AssignNewValue_17^Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp9^Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_1&^Bottleneck1/depthbatch/ReadVariableOp(^Bottleneck1/depthbatch/ReadVariableOp_1-^Bottleneck1/depthconv/BiasAdd/ReadVariableOp/^Bottleneck1/depthconv/depthwise/ReadVariableOp$^Bottleneck1/expbatch/AssignNewValue&^Bottleneck1/expbatch/AssignNewValue_15^Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp7^Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_1$^Bottleneck1/expbatch/ReadVariableOp&^Bottleneck1/expbatch/ReadVariableOp_1+^Bottleneck1/expconv/BiasAdd/ReadVariableOp*^Bottleneck1/expconv/Conv2D/ReadVariableOp%^Bottleneck1/projbatch/AssignNewValue'^Bottleneck1/projbatch/AssignNewValue_16^Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp8^Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_1%^Bottleneck1/projbatch/ReadVariableOp'^Bottleneck1/projbatch/ReadVariableOp_1,^Bottleneck1/projconv/BiasAdd/ReadVariableOp+^Bottleneck1/projconv/Conv2D/ReadVariableOp&^Bottleneck1/shortbatch/AssignNewValue(^Bottleneck1/shortbatch/AssignNewValue_17^Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp9^Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_1&^Bottleneck1/shortbatch/ReadVariableOp(^Bottleneck1/shortbatch/ReadVariableOp_1&^Bottleneck2/depthbatch/AssignNewValue(^Bottleneck2/depthbatch/AssignNewValue_17^Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp9^Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_1&^Bottleneck2/depthbatch/ReadVariableOp(^Bottleneck2/depthbatch/ReadVariableOp_1-^Bottleneck2/depthconv/BiasAdd/ReadVariableOp/^Bottleneck2/depthconv/depthwise/ReadVariableOp$^Bottleneck2/expbatch/AssignNewValue&^Bottleneck2/expbatch/AssignNewValue_15^Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp7^Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_1$^Bottleneck2/expbatch/ReadVariableOp&^Bottleneck2/expbatch/ReadVariableOp_1+^Bottleneck2/expconv/BiasAdd/ReadVariableOp*^Bottleneck2/expconv/Conv2D/ReadVariableOp%^Bottleneck2/projbatch/AssignNewValue'^Bottleneck2/projbatch/AssignNewValue_16^Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp8^Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_1%^Bottleneck2/projbatch/ReadVariableOp'^Bottleneck2/projbatch/ReadVariableOp_1,^Bottleneck2/projconv/BiasAdd/ReadVariableOp+^Bottleneck2/projconv/Conv2D/ReadVariableOp&^Bottleneck2/shortbatch/AssignNewValue(^Bottleneck2/shortbatch/AssignNewValue_17^Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp9^Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_1&^Bottleneck2/shortbatch/ReadVariableOp(^Bottleneck2/shortbatch/ReadVariableOp_1&^batch_normalization_34/AssignNewValue(^batch_normalization_34/AssignNewValue_17^batch_normalization_34/FusedBatchNormV3/ReadVariableOp9^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_34/ReadVariableOp(^batch_normalization_34/ReadVariableOp_1!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџрр: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%Bottleneck1/depthbatch/AssignNewValue%Bottleneck1/depthbatch/AssignNewValue2R
'Bottleneck1/depthbatch/AssignNewValue_1'Bottleneck1/depthbatch/AssignNewValue_12p
6Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp6Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp2t
8Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_18Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_12N
%Bottleneck1/depthbatch/ReadVariableOp%Bottleneck1/depthbatch/ReadVariableOp2R
'Bottleneck1/depthbatch/ReadVariableOp_1'Bottleneck1/depthbatch/ReadVariableOp_12\
,Bottleneck1/depthconv/BiasAdd/ReadVariableOp,Bottleneck1/depthconv/BiasAdd/ReadVariableOp2`
.Bottleneck1/depthconv/depthwise/ReadVariableOp.Bottleneck1/depthconv/depthwise/ReadVariableOp2J
#Bottleneck1/expbatch/AssignNewValue#Bottleneck1/expbatch/AssignNewValue2N
%Bottleneck1/expbatch/AssignNewValue_1%Bottleneck1/expbatch/AssignNewValue_12l
4Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp4Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp2p
6Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_16Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_12J
#Bottleneck1/expbatch/ReadVariableOp#Bottleneck1/expbatch/ReadVariableOp2N
%Bottleneck1/expbatch/ReadVariableOp_1%Bottleneck1/expbatch/ReadVariableOp_12X
*Bottleneck1/expconv/BiasAdd/ReadVariableOp*Bottleneck1/expconv/BiasAdd/ReadVariableOp2V
)Bottleneck1/expconv/Conv2D/ReadVariableOp)Bottleneck1/expconv/Conv2D/ReadVariableOp2L
$Bottleneck1/projbatch/AssignNewValue$Bottleneck1/projbatch/AssignNewValue2P
&Bottleneck1/projbatch/AssignNewValue_1&Bottleneck1/projbatch/AssignNewValue_12n
5Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp5Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp2r
7Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_17Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_12L
$Bottleneck1/projbatch/ReadVariableOp$Bottleneck1/projbatch/ReadVariableOp2P
&Bottleneck1/projbatch/ReadVariableOp_1&Bottleneck1/projbatch/ReadVariableOp_12Z
+Bottleneck1/projconv/BiasAdd/ReadVariableOp+Bottleneck1/projconv/BiasAdd/ReadVariableOp2X
*Bottleneck1/projconv/Conv2D/ReadVariableOp*Bottleneck1/projconv/Conv2D/ReadVariableOp2N
%Bottleneck1/shortbatch/AssignNewValue%Bottleneck1/shortbatch/AssignNewValue2R
'Bottleneck1/shortbatch/AssignNewValue_1'Bottleneck1/shortbatch/AssignNewValue_12p
6Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp6Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp2t
8Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_18Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_12N
%Bottleneck1/shortbatch/ReadVariableOp%Bottleneck1/shortbatch/ReadVariableOp2R
'Bottleneck1/shortbatch/ReadVariableOp_1'Bottleneck1/shortbatch/ReadVariableOp_12N
%Bottleneck2/depthbatch/AssignNewValue%Bottleneck2/depthbatch/AssignNewValue2R
'Bottleneck2/depthbatch/AssignNewValue_1'Bottleneck2/depthbatch/AssignNewValue_12p
6Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp6Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp2t
8Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_18Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_12N
%Bottleneck2/depthbatch/ReadVariableOp%Bottleneck2/depthbatch/ReadVariableOp2R
'Bottleneck2/depthbatch/ReadVariableOp_1'Bottleneck2/depthbatch/ReadVariableOp_12\
,Bottleneck2/depthconv/BiasAdd/ReadVariableOp,Bottleneck2/depthconv/BiasAdd/ReadVariableOp2`
.Bottleneck2/depthconv/depthwise/ReadVariableOp.Bottleneck2/depthconv/depthwise/ReadVariableOp2J
#Bottleneck2/expbatch/AssignNewValue#Bottleneck2/expbatch/AssignNewValue2N
%Bottleneck2/expbatch/AssignNewValue_1%Bottleneck2/expbatch/AssignNewValue_12l
4Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp4Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp2p
6Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_16Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_12J
#Bottleneck2/expbatch/ReadVariableOp#Bottleneck2/expbatch/ReadVariableOp2N
%Bottleneck2/expbatch/ReadVariableOp_1%Bottleneck2/expbatch/ReadVariableOp_12X
*Bottleneck2/expconv/BiasAdd/ReadVariableOp*Bottleneck2/expconv/BiasAdd/ReadVariableOp2V
)Bottleneck2/expconv/Conv2D/ReadVariableOp)Bottleneck2/expconv/Conv2D/ReadVariableOp2L
$Bottleneck2/projbatch/AssignNewValue$Bottleneck2/projbatch/AssignNewValue2P
&Bottleneck2/projbatch/AssignNewValue_1&Bottleneck2/projbatch/AssignNewValue_12n
5Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp5Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp2r
7Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_17Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_12L
$Bottleneck2/projbatch/ReadVariableOp$Bottleneck2/projbatch/ReadVariableOp2P
&Bottleneck2/projbatch/ReadVariableOp_1&Bottleneck2/projbatch/ReadVariableOp_12Z
+Bottleneck2/projconv/BiasAdd/ReadVariableOp+Bottleneck2/projconv/BiasAdd/ReadVariableOp2X
*Bottleneck2/projconv/Conv2D/ReadVariableOp*Bottleneck2/projconv/Conv2D/ReadVariableOp2N
%Bottleneck2/shortbatch/AssignNewValue%Bottleneck2/shortbatch/AssignNewValue2R
'Bottleneck2/shortbatch/AssignNewValue_1'Bottleneck2/shortbatch/AssignNewValue_12p
6Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp6Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp2t
8Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_18Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_12N
%Bottleneck2/shortbatch/ReadVariableOp%Bottleneck2/shortbatch/ReadVariableOp2R
'Bottleneck2/shortbatch/ReadVariableOp_1'Bottleneck2/shortbatch/ReadVariableOp_12N
%batch_normalization_34/AssignNewValue%batch_normalization_34/AssignNewValue2R
'batch_normalization_34/AssignNewValue_1'batch_normalization_34/AssignNewValue_12p
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp6batch_normalization_34/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_18batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_34/ReadVariableOp%batch_normalization_34/ReadVariableOp2R
'batch_normalization_34/ReadVariableOp_1'batch_normalization_34/ReadVariableOp_12D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџрр
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_47256

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Н
E
)__inference_dropout_8_layer_call_fn_50920

inputs
identityЗ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_48127h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Н
E
)__inference_dropout_7_layer_call_fn_50617

inputs
identityЗ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_47989h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
љ
Г
D__inference_projbatch_layer_call_and_return_conditional_losses_51407

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Р

E__inference_shortbatch_layer_call_and_return_conditional_losses_47473

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Р

E__inference_depthbatch_layer_call_and_return_conditional_losses_47345

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ј
В
C__inference_expbatch_layer_call_and_return_conditional_losses_47580

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
љ
Х
*__inference_depthbatch_layer_call_fn_51048

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_depthbatch_layer_call_and_return_conditional_losses_47345
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Њі
@
__inference__traced_save_51882
file_prefix/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop;
7savev2_batch_normalization_34_gamma_read_readvariableop:
6savev2_batch_normalization_34_beta_read_readvariableopA
=savev2_batch_normalization_34_moving_mean_read_readvariableopE
Asavev2_batch_normalization_34_moving_variance_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop9
5savev2_bottleneck1_expconv_kernel_read_readvariableop7
3savev2_bottleneck1_expconv_bias_read_readvariableop9
5savev2_bottleneck1_expbatch_gamma_read_readvariableop8
4savev2_bottleneck1_expbatch_beta_read_readvariableopE
Asavev2_bottleneck1_depthconv_depthwise_kernel_read_readvariableop9
5savev2_bottleneck1_depthconv_bias_read_readvariableop;
7savev2_bottleneck1_depthbatch_gamma_read_readvariableop:
6savev2_bottleneck1_depthbatch_beta_read_readvariableop:
6savev2_bottleneck1_projconv_kernel_read_readvariableop8
4savev2_bottleneck1_projconv_bias_read_readvariableop:
6savev2_bottleneck1_projbatch_gamma_read_readvariableop9
5savev2_bottleneck1_projbatch_beta_read_readvariableop;
7savev2_bottleneck1_shortbatch_gamma_read_readvariableop:
6savev2_bottleneck1_shortbatch_beta_read_readvariableop?
;savev2_bottleneck1_expbatch_moving_mean_read_readvariableopC
?savev2_bottleneck1_expbatch_moving_variance_read_readvariableopA
=savev2_bottleneck1_depthbatch_moving_mean_read_readvariableopE
Asavev2_bottleneck1_depthbatch_moving_variance_read_readvariableop@
<savev2_bottleneck1_projbatch_moving_mean_read_readvariableopD
@savev2_bottleneck1_projbatch_moving_variance_read_readvariableopA
=savev2_bottleneck1_shortbatch_moving_mean_read_readvariableopE
Asavev2_bottleneck1_shortbatch_moving_variance_read_readvariableop9
5savev2_bottleneck2_expconv_kernel_read_readvariableop7
3savev2_bottleneck2_expconv_bias_read_readvariableop9
5savev2_bottleneck2_expbatch_gamma_read_readvariableop8
4savev2_bottleneck2_expbatch_beta_read_readvariableopE
Asavev2_bottleneck2_depthconv_depthwise_kernel_read_readvariableop9
5savev2_bottleneck2_depthconv_bias_read_readvariableop;
7savev2_bottleneck2_depthbatch_gamma_read_readvariableop:
6savev2_bottleneck2_depthbatch_beta_read_readvariableop:
6savev2_bottleneck2_projconv_kernel_read_readvariableop8
4savev2_bottleneck2_projconv_bias_read_readvariableop:
6savev2_bottleneck2_projbatch_gamma_read_readvariableop9
5savev2_bottleneck2_projbatch_beta_read_readvariableop;
7savev2_bottleneck2_shortbatch_gamma_read_readvariableop:
6savev2_bottleneck2_shortbatch_beta_read_readvariableop?
;savev2_bottleneck2_expbatch_moving_mean_read_readvariableopC
?savev2_bottleneck2_expbatch_moving_variance_read_readvariableopA
=savev2_bottleneck2_depthbatch_moving_mean_read_readvariableopE
Asavev2_bottleneck2_depthbatch_moving_variance_read_readvariableop@
<savev2_bottleneck2_projbatch_moving_mean_read_readvariableopD
@savev2_bottleneck2_projbatch_moving_variance_read_readvariableopA
=savev2_bottleneck2_shortbatch_moving_mean_read_readvariableopE
Asavev2_bottleneck2_shortbatch_moving_variance_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_18_kernel_m_read_readvariableop4
0savev2_adam_conv2d_18_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_34_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_34_beta_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop@
<savev2_adam_bottleneck1_expconv_kernel_m_read_readvariableop>
:savev2_adam_bottleneck1_expconv_bias_m_read_readvariableop@
<savev2_adam_bottleneck1_expbatch_gamma_m_read_readvariableop?
;savev2_adam_bottleneck1_expbatch_beta_m_read_readvariableopL
Hsavev2_adam_bottleneck1_depthconv_depthwise_kernel_m_read_readvariableop@
<savev2_adam_bottleneck1_depthconv_bias_m_read_readvariableopB
>savev2_adam_bottleneck1_depthbatch_gamma_m_read_readvariableopA
=savev2_adam_bottleneck1_depthbatch_beta_m_read_readvariableopA
=savev2_adam_bottleneck1_projconv_kernel_m_read_readvariableop?
;savev2_adam_bottleneck1_projconv_bias_m_read_readvariableopA
=savev2_adam_bottleneck1_projbatch_gamma_m_read_readvariableop@
<savev2_adam_bottleneck1_projbatch_beta_m_read_readvariableopB
>savev2_adam_bottleneck1_shortbatch_gamma_m_read_readvariableopA
=savev2_adam_bottleneck1_shortbatch_beta_m_read_readvariableop@
<savev2_adam_bottleneck2_expconv_kernel_m_read_readvariableop>
:savev2_adam_bottleneck2_expconv_bias_m_read_readvariableop@
<savev2_adam_bottleneck2_expbatch_gamma_m_read_readvariableop?
;savev2_adam_bottleneck2_expbatch_beta_m_read_readvariableopL
Hsavev2_adam_bottleneck2_depthconv_depthwise_kernel_m_read_readvariableop@
<savev2_adam_bottleneck2_depthconv_bias_m_read_readvariableopB
>savev2_adam_bottleneck2_depthbatch_gamma_m_read_readvariableopA
=savev2_adam_bottleneck2_depthbatch_beta_m_read_readvariableopA
=savev2_adam_bottleneck2_projconv_kernel_m_read_readvariableop?
;savev2_adam_bottleneck2_projconv_bias_m_read_readvariableopA
=savev2_adam_bottleneck2_projbatch_gamma_m_read_readvariableop@
<savev2_adam_bottleneck2_projbatch_beta_m_read_readvariableopB
>savev2_adam_bottleneck2_shortbatch_gamma_m_read_readvariableopA
=savev2_adam_bottleneck2_shortbatch_beta_m_read_readvariableop6
2savev2_adam_conv2d_18_kernel_v_read_readvariableop4
0savev2_adam_conv2d_18_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_34_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_34_beta_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop@
<savev2_adam_bottleneck1_expconv_kernel_v_read_readvariableop>
:savev2_adam_bottleneck1_expconv_bias_v_read_readvariableop@
<savev2_adam_bottleneck1_expbatch_gamma_v_read_readvariableop?
;savev2_adam_bottleneck1_expbatch_beta_v_read_readvariableopL
Hsavev2_adam_bottleneck1_depthconv_depthwise_kernel_v_read_readvariableop@
<savev2_adam_bottleneck1_depthconv_bias_v_read_readvariableopB
>savev2_adam_bottleneck1_depthbatch_gamma_v_read_readvariableopA
=savev2_adam_bottleneck1_depthbatch_beta_v_read_readvariableopA
=savev2_adam_bottleneck1_projconv_kernel_v_read_readvariableop?
;savev2_adam_bottleneck1_projconv_bias_v_read_readvariableopA
=savev2_adam_bottleneck1_projbatch_gamma_v_read_readvariableop@
<savev2_adam_bottleneck1_projbatch_beta_v_read_readvariableopB
>savev2_adam_bottleneck1_shortbatch_gamma_v_read_readvariableopA
=savev2_adam_bottleneck1_shortbatch_beta_v_read_readvariableop@
<savev2_adam_bottleneck2_expconv_kernel_v_read_readvariableop>
:savev2_adam_bottleneck2_expconv_bias_v_read_readvariableop@
<savev2_adam_bottleneck2_expbatch_gamma_v_read_readvariableop?
;savev2_adam_bottleneck2_expbatch_beta_v_read_readvariableopL
Hsavev2_adam_bottleneck2_depthconv_depthwise_kernel_v_read_readvariableop@
<savev2_adam_bottleneck2_depthconv_bias_v_read_readvariableopB
>savev2_adam_bottleneck2_depthbatch_gamma_v_read_readvariableopA
=savev2_adam_bottleneck2_depthbatch_beta_v_read_readvariableopA
=savev2_adam_bottleneck2_projconv_kernel_v_read_readvariableop?
;savev2_adam_bottleneck2_projconv_bias_v_read_readvariableopA
=savev2_adam_bottleneck2_projbatch_gamma_v_read_readvariableop@
<savev2_adam_bottleneck2_projbatch_beta_v_read_readvariableopB
>savev2_adam_bottleneck2_shortbatch_gamma_v_read_readvariableopA
=savev2_adam_bottleneck2_shortbatch_beta_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Т<
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*ъ;
valueр;Bн;B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHј
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*
valueBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ь=
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableop7savev2_batch_normalization_34_gamma_read_readvariableop6savev2_batch_normalization_34_beta_read_readvariableop=savev2_batch_normalization_34_moving_mean_read_readvariableopAsavev2_batch_normalization_34_moving_variance_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop5savev2_bottleneck1_expconv_kernel_read_readvariableop3savev2_bottleneck1_expconv_bias_read_readvariableop5savev2_bottleneck1_expbatch_gamma_read_readvariableop4savev2_bottleneck1_expbatch_beta_read_readvariableopAsavev2_bottleneck1_depthconv_depthwise_kernel_read_readvariableop5savev2_bottleneck1_depthconv_bias_read_readvariableop7savev2_bottleneck1_depthbatch_gamma_read_readvariableop6savev2_bottleneck1_depthbatch_beta_read_readvariableop6savev2_bottleneck1_projconv_kernel_read_readvariableop4savev2_bottleneck1_projconv_bias_read_readvariableop6savev2_bottleneck1_projbatch_gamma_read_readvariableop5savev2_bottleneck1_projbatch_beta_read_readvariableop7savev2_bottleneck1_shortbatch_gamma_read_readvariableop6savev2_bottleneck1_shortbatch_beta_read_readvariableop;savev2_bottleneck1_expbatch_moving_mean_read_readvariableop?savev2_bottleneck1_expbatch_moving_variance_read_readvariableop=savev2_bottleneck1_depthbatch_moving_mean_read_readvariableopAsavev2_bottleneck1_depthbatch_moving_variance_read_readvariableop<savev2_bottleneck1_projbatch_moving_mean_read_readvariableop@savev2_bottleneck1_projbatch_moving_variance_read_readvariableop=savev2_bottleneck1_shortbatch_moving_mean_read_readvariableopAsavev2_bottleneck1_shortbatch_moving_variance_read_readvariableop5savev2_bottleneck2_expconv_kernel_read_readvariableop3savev2_bottleneck2_expconv_bias_read_readvariableop5savev2_bottleneck2_expbatch_gamma_read_readvariableop4savev2_bottleneck2_expbatch_beta_read_readvariableopAsavev2_bottleneck2_depthconv_depthwise_kernel_read_readvariableop5savev2_bottleneck2_depthconv_bias_read_readvariableop7savev2_bottleneck2_depthbatch_gamma_read_readvariableop6savev2_bottleneck2_depthbatch_beta_read_readvariableop6savev2_bottleneck2_projconv_kernel_read_readvariableop4savev2_bottleneck2_projconv_bias_read_readvariableop6savev2_bottleneck2_projbatch_gamma_read_readvariableop5savev2_bottleneck2_projbatch_beta_read_readvariableop7savev2_bottleneck2_shortbatch_gamma_read_readvariableop6savev2_bottleneck2_shortbatch_beta_read_readvariableop;savev2_bottleneck2_expbatch_moving_mean_read_readvariableop?savev2_bottleneck2_expbatch_moving_variance_read_readvariableop=savev2_bottleneck2_depthbatch_moving_mean_read_readvariableopAsavev2_bottleneck2_depthbatch_moving_variance_read_readvariableop<savev2_bottleneck2_projbatch_moving_mean_read_readvariableop@savev2_bottleneck2_projbatch_moving_variance_read_readvariableop=savev2_bottleneck2_shortbatch_moving_mean_read_readvariableopAsavev2_bottleneck2_shortbatch_moving_variance_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_18_kernel_m_read_readvariableop0savev2_adam_conv2d_18_bias_m_read_readvariableop>savev2_adam_batch_normalization_34_gamma_m_read_readvariableop=savev2_adam_batch_normalization_34_beta_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop<savev2_adam_bottleneck1_expconv_kernel_m_read_readvariableop:savev2_adam_bottleneck1_expconv_bias_m_read_readvariableop<savev2_adam_bottleneck1_expbatch_gamma_m_read_readvariableop;savev2_adam_bottleneck1_expbatch_beta_m_read_readvariableopHsavev2_adam_bottleneck1_depthconv_depthwise_kernel_m_read_readvariableop<savev2_adam_bottleneck1_depthconv_bias_m_read_readvariableop>savev2_adam_bottleneck1_depthbatch_gamma_m_read_readvariableop=savev2_adam_bottleneck1_depthbatch_beta_m_read_readvariableop=savev2_adam_bottleneck1_projconv_kernel_m_read_readvariableop;savev2_adam_bottleneck1_projconv_bias_m_read_readvariableop=savev2_adam_bottleneck1_projbatch_gamma_m_read_readvariableop<savev2_adam_bottleneck1_projbatch_beta_m_read_readvariableop>savev2_adam_bottleneck1_shortbatch_gamma_m_read_readvariableop=savev2_adam_bottleneck1_shortbatch_beta_m_read_readvariableop<savev2_adam_bottleneck2_expconv_kernel_m_read_readvariableop:savev2_adam_bottleneck2_expconv_bias_m_read_readvariableop<savev2_adam_bottleneck2_expbatch_gamma_m_read_readvariableop;savev2_adam_bottleneck2_expbatch_beta_m_read_readvariableopHsavev2_adam_bottleneck2_depthconv_depthwise_kernel_m_read_readvariableop<savev2_adam_bottleneck2_depthconv_bias_m_read_readvariableop>savev2_adam_bottleneck2_depthbatch_gamma_m_read_readvariableop=savev2_adam_bottleneck2_depthbatch_beta_m_read_readvariableop=savev2_adam_bottleneck2_projconv_kernel_m_read_readvariableop;savev2_adam_bottleneck2_projconv_bias_m_read_readvariableop=savev2_adam_bottleneck2_projbatch_gamma_m_read_readvariableop<savev2_adam_bottleneck2_projbatch_beta_m_read_readvariableop>savev2_adam_bottleneck2_shortbatch_gamma_m_read_readvariableop=savev2_adam_bottleneck2_shortbatch_beta_m_read_readvariableop2savev2_adam_conv2d_18_kernel_v_read_readvariableop0savev2_adam_conv2d_18_bias_v_read_readvariableop>savev2_adam_batch_normalization_34_gamma_v_read_readvariableop=savev2_adam_batch_normalization_34_beta_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop<savev2_adam_bottleneck1_expconv_kernel_v_read_readvariableop:savev2_adam_bottleneck1_expconv_bias_v_read_readvariableop<savev2_adam_bottleneck1_expbatch_gamma_v_read_readvariableop;savev2_adam_bottleneck1_expbatch_beta_v_read_readvariableopHsavev2_adam_bottleneck1_depthconv_depthwise_kernel_v_read_readvariableop<savev2_adam_bottleneck1_depthconv_bias_v_read_readvariableop>savev2_adam_bottleneck1_depthbatch_gamma_v_read_readvariableop=savev2_adam_bottleneck1_depthbatch_beta_v_read_readvariableop=savev2_adam_bottleneck1_projconv_kernel_v_read_readvariableop;savev2_adam_bottleneck1_projconv_bias_v_read_readvariableop=savev2_adam_bottleneck1_projbatch_gamma_v_read_readvariableop<savev2_adam_bottleneck1_projbatch_beta_v_read_readvariableop>savev2_adam_bottleneck1_shortbatch_gamma_v_read_readvariableop=savev2_adam_bottleneck1_shortbatch_beta_v_read_readvariableop<savev2_adam_bottleneck2_expconv_kernel_v_read_readvariableop:savev2_adam_bottleneck2_expconv_bias_v_read_readvariableop<savev2_adam_bottleneck2_expbatch_gamma_v_read_readvariableop;savev2_adam_bottleneck2_expbatch_beta_v_read_readvariableopHsavev2_adam_bottleneck2_depthconv_depthwise_kernel_v_read_readvariableop<savev2_adam_bottleneck2_depthconv_bias_v_read_readvariableop>savev2_adam_bottleneck2_depthbatch_gamma_v_read_readvariableop=savev2_adam_bottleneck2_depthbatch_beta_v_read_readvariableop=savev2_adam_bottleneck2_projconv_kernel_v_read_readvariableop;savev2_adam_bottleneck2_projconv_bias_v_read_readvariableop=savev2_adam_bottleneck2_projbatch_gamma_v_read_readvariableop<savev2_adam_bottleneck2_projbatch_beta_v_read_readvariableop>savev2_adam_bottleneck2_shortbatch_gamma_v_read_readvariableop=savev2_adam_bottleneck2_shortbatch_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesі
ѓ: :::::::		::::::::::::::::::::::::0:0:0:0:0:0:0:0:0::::::0:0:0:0::::: : : : : : : : : : :::::		::::::::::::::::0:0:0:0:0:0:0:0:0::::::::::		::::::::::::::::0:0:0:0:0:0:0:0:0:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:		: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:0:  

_output_shapes
:0: !

_output_shapes
:0: "

_output_shapes
:0:,#(
&
_output_shapes
:0: $

_output_shapes
:0: %

_output_shapes
:0: &

_output_shapes
:0:,'(
&
_output_shapes
:0: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:0: .

_output_shapes
:0: /

_output_shapes
:0: 0

_output_shapes
:0: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
::5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :,?(
&
_output_shapes
:: @

_output_shapes
:: A

_output_shapes
:: B

_output_shapes
::%C!

_output_shapes
:		: D

_output_shapes
::,E(
&
_output_shapes
:: F

_output_shapes
:: G

_output_shapes
:: H

_output_shapes
::,I(
&
_output_shapes
:: J

_output_shapes
:: K

_output_shapes
:: L

_output_shapes
::,M(
&
_output_shapes
:: N

_output_shapes
:: O

_output_shapes
:: P

_output_shapes
:: Q

_output_shapes
:: R

_output_shapes
::,S(
&
_output_shapes
:0: T

_output_shapes
:0: U

_output_shapes
:0: V

_output_shapes
:0:,W(
&
_output_shapes
:0: X

_output_shapes
:0: Y

_output_shapes
:0: Z

_output_shapes
:0:,[(
&
_output_shapes
:0: \

_output_shapes
:: ]

_output_shapes
:: ^

_output_shapes
:: _

_output_shapes
:: `

_output_shapes
::,a(
&
_output_shapes
:: b

_output_shapes
:: c

_output_shapes
:: d

_output_shapes
::%e!

_output_shapes
:		: f

_output_shapes
::,g(
&
_output_shapes
:: h

_output_shapes
:: i

_output_shapes
:: j

_output_shapes
::,k(
&
_output_shapes
:: l

_output_shapes
:: m

_output_shapes
:: n

_output_shapes
::,o(
&
_output_shapes
:: p

_output_shapes
:: q

_output_shapes
:: r

_output_shapes
:: s

_output_shapes
:: t

_output_shapes
::,u(
&
_output_shapes
:0: v

_output_shapes
:0: w

_output_shapes
:0: x

_output_shapes
:0:,y(
&
_output_shapes
:0: z

_output_shapes
:0: {

_output_shapes
:0: |

_output_shapes
:0:,}(
&
_output_shapes
:0: ~

_output_shapes
:: 

_output_shapes
::!

_output_shapes
::!

_output_shapes
::!

_output_shapes
::

_output_shapes
: 
юX

F__inference_Bottleneck2_layer_call_and_return_conditional_losses_48075
x@
&expconv_conv2d_readvariableop_resource:05
'expconv_biasadd_readvariableop_resource:0.
 expbatch_readvariableop_resource:00
"expbatch_readvariableop_1_resource:0?
1expbatch_fusedbatchnormv3_readvariableop_resource:0A
3expbatch_fusedbatchnormv3_readvariableop_1_resource:0E
+depthconv_depthwise_readvariableop_resource:07
)depthconv_biasadd_readvariableop_resource:00
"depthbatch_readvariableop_resource:02
$depthbatch_readvariableop_1_resource:0A
3depthbatch_fusedbatchnormv3_readvariableop_resource:0C
5depthbatch_fusedbatchnormv3_readvariableop_1_resource:0A
'projconv_conv2d_readvariableop_resource:06
(projconv_biasadd_readvariableop_resource:/
!projbatch_readvariableop_resource:1
#projbatch_readvariableop_1_resource:@
2projbatch_fusedbatchnormv3_readvariableop_resource:B
4projbatch_fusedbatchnormv3_readvariableop_1_resource:0
"shortbatch_readvariableop_resource:2
$shortbatch_readvariableop_1_resource:A
3shortbatch_fusedbatchnormv3_readvariableop_resource:C
5shortbatch_fusedbatchnormv3_readvariableop_1_resource:
identityЂ*depthbatch/FusedBatchNormV3/ReadVariableOpЂ,depthbatch/FusedBatchNormV3/ReadVariableOp_1Ђdepthbatch/ReadVariableOpЂdepthbatch/ReadVariableOp_1Ђ depthconv/BiasAdd/ReadVariableOpЂ"depthconv/depthwise/ReadVariableOpЂ(expbatch/FusedBatchNormV3/ReadVariableOpЂ*expbatch/FusedBatchNormV3/ReadVariableOp_1Ђexpbatch/ReadVariableOpЂexpbatch/ReadVariableOp_1Ђexpconv/BiasAdd/ReadVariableOpЂexpconv/Conv2D/ReadVariableOpЂ)projbatch/FusedBatchNormV3/ReadVariableOpЂ+projbatch/FusedBatchNormV3/ReadVariableOp_1Ђprojbatch/ReadVariableOpЂprojbatch/ReadVariableOp_1Ђprojconv/BiasAdd/ReadVariableOpЂprojconv/Conv2D/ReadVariableOpЂ*shortbatch/FusedBatchNormV3/ReadVariableOpЂ,shortbatch/FusedBatchNormV3/ReadVariableOp_1Ђshortbatch/ReadVariableOpЂshortbatch/ReadVariableOp_1
expconv/Conv2D/ReadVariableOpReadVariableOp&expconv_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0Є
expconv/Conv2DConv2Dx%expconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides

expconv/BiasAdd/ReadVariableOpReadVariableOp'expconv_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
expconv/BiasAddBiasAddexpconv/Conv2D:output:0&expconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0t
expbatch/ReadVariableOpReadVariableOp expbatch_readvariableop_resource*
_output_shapes
:0*
dtype0x
expbatch/ReadVariableOp_1ReadVariableOp"expbatch_readvariableop_1_resource*
_output_shapes
:0*
dtype0
(expbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp1expbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
*expbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3expbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0ѕ
expbatch/FusedBatchNormV3FusedBatchNormV3expconv/BiasAdd:output:0expbatch/ReadVariableOp:value:0!expbatch/ReadVariableOp_1:value:00expbatch/FusedBatchNormV3/ReadVariableOp:value:02expbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( p
activation/ReluReluexpbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ0
"depthconv/depthwise/ReadVariableOpReadVariableOp+depthconv_depthwise_readvariableop_resource*&
_output_shapes
:0*
dtype0r
depthconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      r
!depthconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      й
depthconv/depthwiseDepthwiseConv2dNativeactivation/Relu:activations:0*depthconv/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides

 depthconv/BiasAdd/ReadVariableOpReadVariableOp)depthconv_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
depthconv/BiasAddBiasAdddepthconv/depthwise:output:0(depthconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0x
depthbatch/ReadVariableOpReadVariableOp"depthbatch_readvariableop_resource*
_output_shapes
:0*
dtype0|
depthbatch/ReadVariableOp_1ReadVariableOp$depthbatch_readvariableop_1_resource*
_output_shapes
:0*
dtype0
*depthbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp3depthbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
,depthbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5depthbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0
depthbatch/FusedBatchNormV3FusedBatchNormV3depthconv/BiasAdd:output:0!depthbatch/ReadVariableOp:value:0#depthbatch/ReadVariableOp_1:value:02depthbatch/FusedBatchNormV3/ReadVariableOp:value:04depthbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( t
activation_1/ReluReludepthbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ0
projconv/Conv2D/ReadVariableOpReadVariableOp'projconv_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0Ф
projconv/Conv2DConv2Dactivation_1/Relu:activations:0&projconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

projconv/BiasAdd/ReadVariableOpReadVariableOp(projconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
projconv/BiasAddBiasAddprojconv/Conv2D:output:0'projconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџv
projbatch/ReadVariableOpReadVariableOp!projbatch_readvariableop_resource*
_output_shapes
:*
dtype0z
projbatch/ReadVariableOp_1ReadVariableOp#projbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
)projbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp2projbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
+projbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4projbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ћ
projbatch/FusedBatchNormV3FusedBatchNormV3projconv/BiasAdd:output:0 projbatch/ReadVariableOp:value:0"projbatch/ReadVariableOp_1:value:01projbatch/FusedBatchNormV3/ReadVariableOp:value:03projbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
is_training( x
shortbatch/ReadVariableOpReadVariableOp"shortbatch_readvariableop_resource*
_output_shapes
:*
dtype0|
shortbatch/ReadVariableOp_1ReadVariableOp$shortbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
*shortbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp3shortbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
,shortbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5shortbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ш
shortbatch/FusedBatchNormV3FusedBatchNormV3x!shortbatch/ReadVariableOp:value:0#shortbatch/ReadVariableOp_1:value:02shortbatch/FusedBatchNormV3/ReadVariableOp:value:04shortbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 
add/addAddV2projbatch/FusedBatchNormV3:y:0shortbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ`
activation_2/ReluReluadd/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџv
IdentityIdentityactivation_2/Relu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџо
NoOpNoOp+^depthbatch/FusedBatchNormV3/ReadVariableOp-^depthbatch/FusedBatchNormV3/ReadVariableOp_1^depthbatch/ReadVariableOp^depthbatch/ReadVariableOp_1!^depthconv/BiasAdd/ReadVariableOp#^depthconv/depthwise/ReadVariableOp)^expbatch/FusedBatchNormV3/ReadVariableOp+^expbatch/FusedBatchNormV3/ReadVariableOp_1^expbatch/ReadVariableOp^expbatch/ReadVariableOp_1^expconv/BiasAdd/ReadVariableOp^expconv/Conv2D/ReadVariableOp*^projbatch/FusedBatchNormV3/ReadVariableOp,^projbatch/FusedBatchNormV3/ReadVariableOp_1^projbatch/ReadVariableOp^projbatch/ReadVariableOp_1 ^projconv/BiasAdd/ReadVariableOp^projconv/Conv2D/ReadVariableOp+^shortbatch/FusedBatchNormV3/ReadVariableOp-^shortbatch/FusedBatchNormV3/ReadVariableOp_1^shortbatch/ReadVariableOp^shortbatch/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 2X
*depthbatch/FusedBatchNormV3/ReadVariableOp*depthbatch/FusedBatchNormV3/ReadVariableOp2\
,depthbatch/FusedBatchNormV3/ReadVariableOp_1,depthbatch/FusedBatchNormV3/ReadVariableOp_126
depthbatch/ReadVariableOpdepthbatch/ReadVariableOp2:
depthbatch/ReadVariableOp_1depthbatch/ReadVariableOp_12D
 depthconv/BiasAdd/ReadVariableOp depthconv/BiasAdd/ReadVariableOp2H
"depthconv/depthwise/ReadVariableOp"depthconv/depthwise/ReadVariableOp2T
(expbatch/FusedBatchNormV3/ReadVariableOp(expbatch/FusedBatchNormV3/ReadVariableOp2X
*expbatch/FusedBatchNormV3/ReadVariableOp_1*expbatch/FusedBatchNormV3/ReadVariableOp_122
expbatch/ReadVariableOpexpbatch/ReadVariableOp26
expbatch/ReadVariableOp_1expbatch/ReadVariableOp_12@
expconv/BiasAdd/ReadVariableOpexpconv/BiasAdd/ReadVariableOp2>
expconv/Conv2D/ReadVariableOpexpconv/Conv2D/ReadVariableOp2V
)projbatch/FusedBatchNormV3/ReadVariableOp)projbatch/FusedBatchNormV3/ReadVariableOp2Z
+projbatch/FusedBatchNormV3/ReadVariableOp_1+projbatch/FusedBatchNormV3/ReadVariableOp_124
projbatch/ReadVariableOpprojbatch/ReadVariableOp28
projbatch/ReadVariableOp_1projbatch/ReadVariableOp_12B
projconv/BiasAdd/ReadVariableOpprojconv/BiasAdd/ReadVariableOp2@
projconv/Conv2D/ReadVariableOpprojconv/Conv2D/ReadVariableOp2X
*shortbatch/FusedBatchNormV3/ReadVariableOp*shortbatch/FusedBatchNormV3/ReadVariableOp2\
,shortbatch/FusedBatchNormV3/ReadVariableOp_1,shortbatch/FusedBatchNormV3/ReadVariableOp_126
shortbatch/ReadVariableOpshortbatch/ReadVariableOp2:
shortbatch/ReadVariableOp_1shortbatch/ReadVariableOp_1:R N
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameX
У
H
,__inference_activation_2_layer_call_fn_50294

inputs
identityК
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџpp* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_47843h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџpp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџpp:W S
/
_output_shapes
:џџџџџџџџџpp
 
_user_specified_nameinputs

b
)__inference_dropout_6_layer_call_fn_50319

inputs
identityЂStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ88* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_48710w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ88`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ8822
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ88
 
_user_specified_nameinputs

Р
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_47236

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ы
c
G__inference_activation_2_layer_call_and_return_conditional_losses_47843

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџppb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџpp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџpp:W S
/
_output_shapes
:џџџџџџџџџpp
 
_user_specified_nameinputs
	
б
6__inference_batch_normalization_34_layer_call_fn_50240

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_47205
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ј
b
F__inference_rescaling_2_layer_call_and_return_conditional_losses_50208

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџррd
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџррY
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:џџџџџџџџџрр"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџрр:Y U
1
_output_shapes
:џџџџџџџџџрр
 
_user_specified_nameinputs
	
б
6__inference_batch_normalization_34_layer_call_fn_50253

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_47236
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ю

)__inference_conv2d_18_layer_call_fn_50217

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџpp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_47823w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџpp`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџрр: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџрр
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_47524

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
цo
њ
F__inference_Bottleneck1_layer_call_and_return_conditional_losses_48643
x@
&expconv_conv2d_readvariableop_resource:5
'expconv_biasadd_readvariableop_resource:.
 expbatch_readvariableop_resource:0
"expbatch_readvariableop_1_resource:?
1expbatch_fusedbatchnormv3_readvariableop_resource:A
3expbatch_fusedbatchnormv3_readvariableop_1_resource:E
+depthconv_depthwise_readvariableop_resource:7
)depthconv_biasadd_readvariableop_resource:0
"depthbatch_readvariableop_resource:2
$depthbatch_readvariableop_1_resource:A
3depthbatch_fusedbatchnormv3_readvariableop_resource:C
5depthbatch_fusedbatchnormv3_readvariableop_1_resource:A
'projconv_conv2d_readvariableop_resource:6
(projconv_biasadd_readvariableop_resource:/
!projbatch_readvariableop_resource:1
#projbatch_readvariableop_1_resource:@
2projbatch_fusedbatchnormv3_readvariableop_resource:B
4projbatch_fusedbatchnormv3_readvariableop_1_resource:0
"shortbatch_readvariableop_resource:2
$shortbatch_readvariableop_1_resource:A
3shortbatch_fusedbatchnormv3_readvariableop_resource:C
5shortbatch_fusedbatchnormv3_readvariableop_1_resource:
identityЂdepthbatch/AssignNewValueЂdepthbatch/AssignNewValue_1Ђ*depthbatch/FusedBatchNormV3/ReadVariableOpЂ,depthbatch/FusedBatchNormV3/ReadVariableOp_1Ђdepthbatch/ReadVariableOpЂdepthbatch/ReadVariableOp_1Ђ depthconv/BiasAdd/ReadVariableOpЂ"depthconv/depthwise/ReadVariableOpЂexpbatch/AssignNewValueЂexpbatch/AssignNewValue_1Ђ(expbatch/FusedBatchNormV3/ReadVariableOpЂ*expbatch/FusedBatchNormV3/ReadVariableOp_1Ђexpbatch/ReadVariableOpЂexpbatch/ReadVariableOp_1Ђexpconv/BiasAdd/ReadVariableOpЂexpconv/Conv2D/ReadVariableOpЂprojbatch/AssignNewValueЂprojbatch/AssignNewValue_1Ђ)projbatch/FusedBatchNormV3/ReadVariableOpЂ+projbatch/FusedBatchNormV3/ReadVariableOp_1Ђprojbatch/ReadVariableOpЂprojbatch/ReadVariableOp_1Ђprojconv/BiasAdd/ReadVariableOpЂprojconv/Conv2D/ReadVariableOpЂshortbatch/AssignNewValueЂshortbatch/AssignNewValue_1Ђ*shortbatch/FusedBatchNormV3/ReadVariableOpЂ,shortbatch/FusedBatchNormV3/ReadVariableOp_1Ђshortbatch/ReadVariableOpЂshortbatch/ReadVariableOp_1
expconv/Conv2D/ReadVariableOpReadVariableOp&expconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Є
expconv/Conv2DConv2Dx%expconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

expconv/BiasAdd/ReadVariableOpReadVariableOp'expconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
expconv/BiasAddBiasAddexpconv/Conv2D:output:0&expconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88t
expbatch/ReadVariableOpReadVariableOp expbatch_readvariableop_resource*
_output_shapes
:*
dtype0x
expbatch/ReadVariableOp_1ReadVariableOp"expbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
(expbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp1expbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
*expbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3expbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
expbatch/FusedBatchNormV3FusedBatchNormV3expconv/BiasAdd:output:0expbatch/ReadVariableOp:value:0!expbatch/ReadVariableOp_1:value:00expbatch/FusedBatchNormV3/ReadVariableOp:value:02expbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
exponential_avg_factor%
з#<ъ
expbatch/AssignNewValueAssignVariableOp1expbatch_fusedbatchnormv3_readvariableop_resource&expbatch/FusedBatchNormV3:batch_mean:0)^expbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(є
expbatch/AssignNewValue_1AssignVariableOp3expbatch_fusedbatchnormv3_readvariableop_1_resource*expbatch/FusedBatchNormV3:batch_variance:0+^expbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(p
activation/ReluReluexpbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88
"depthconv/depthwise/ReadVariableOpReadVariableOp+depthconv_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0r
depthconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            r
!depthconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      й
depthconv/depthwiseDepthwiseConv2dNativeactivation/Relu:activations:0*depthconv/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

 depthconv/BiasAdd/ReadVariableOpReadVariableOp)depthconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
depthconv/BiasAddBiasAdddepthconv/depthwise:output:0(depthconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88x
depthbatch/ReadVariableOpReadVariableOp"depthbatch_readvariableop_resource*
_output_shapes
:*
dtype0|
depthbatch/ReadVariableOp_1ReadVariableOp$depthbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
*depthbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp3depthbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
,depthbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5depthbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
depthbatch/FusedBatchNormV3FusedBatchNormV3depthconv/BiasAdd:output:0!depthbatch/ReadVariableOp:value:0#depthbatch/ReadVariableOp_1:value:02depthbatch/FusedBatchNormV3/ReadVariableOp:value:04depthbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
exponential_avg_factor%
з#<ђ
depthbatch/AssignNewValueAssignVariableOp3depthbatch_fusedbatchnormv3_readvariableop_resource(depthbatch/FusedBatchNormV3:batch_mean:0+^depthbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ќ
depthbatch/AssignNewValue_1AssignVariableOp5depthbatch_fusedbatchnormv3_readvariableop_1_resource,depthbatch/FusedBatchNormV3:batch_variance:0-^depthbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(t
activation_1/ReluReludepthbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88
projconv/Conv2D/ReadVariableOpReadVariableOp'projconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
projconv/Conv2DConv2Dactivation_1/Relu:activations:0&projconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

projconv/BiasAdd/ReadVariableOpReadVariableOp(projconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
projconv/BiasAddBiasAddprojconv/Conv2D:output:0'projconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88v
projbatch/ReadVariableOpReadVariableOp!projbatch_readvariableop_resource*
_output_shapes
:*
dtype0z
projbatch/ReadVariableOp_1ReadVariableOp#projbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
)projbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp2projbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
+projbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4projbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
projbatch/FusedBatchNormV3FusedBatchNormV3projconv/BiasAdd:output:0 projbatch/ReadVariableOp:value:0"projbatch/ReadVariableOp_1:value:01projbatch/FusedBatchNormV3/ReadVariableOp:value:03projbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
exponential_avg_factor%
з#<ю
projbatch/AssignNewValueAssignVariableOp2projbatch_fusedbatchnormv3_readvariableop_resource'projbatch/FusedBatchNormV3:batch_mean:0*^projbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ј
projbatch/AssignNewValue_1AssignVariableOp4projbatch_fusedbatchnormv3_readvariableop_1_resource+projbatch/FusedBatchNormV3:batch_variance:0,^projbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(x
shortbatch/ReadVariableOpReadVariableOp"shortbatch_readvariableop_resource*
_output_shapes
:*
dtype0|
shortbatch/ReadVariableOp_1ReadVariableOp$shortbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
*shortbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp3shortbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
,shortbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5shortbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0і
shortbatch/FusedBatchNormV3FusedBatchNormV3x!shortbatch/ReadVariableOp:value:0#shortbatch/ReadVariableOp_1:value:02shortbatch/FusedBatchNormV3/ReadVariableOp:value:04shortbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
exponential_avg_factor%
з#<ђ
shortbatch/AssignNewValueAssignVariableOp3shortbatch_fusedbatchnormv3_readvariableop_resource(shortbatch/FusedBatchNormV3:batch_mean:0+^shortbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ќ
shortbatch/AssignNewValue_1AssignVariableOp5shortbatch_fusedbatchnormv3_readvariableop_1_resource,shortbatch/FusedBatchNormV3:batch_variance:0-^shortbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
add/addAddV2projbatch/FusedBatchNormV3:y:0shortbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88`
activation_2/ReluReluadd/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ88v
IdentityIdentityactivation_2/Relu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ88Р
NoOpNoOp^depthbatch/AssignNewValue^depthbatch/AssignNewValue_1+^depthbatch/FusedBatchNormV3/ReadVariableOp-^depthbatch/FusedBatchNormV3/ReadVariableOp_1^depthbatch/ReadVariableOp^depthbatch/ReadVariableOp_1!^depthconv/BiasAdd/ReadVariableOp#^depthconv/depthwise/ReadVariableOp^expbatch/AssignNewValue^expbatch/AssignNewValue_1)^expbatch/FusedBatchNormV3/ReadVariableOp+^expbatch/FusedBatchNormV3/ReadVariableOp_1^expbatch/ReadVariableOp^expbatch/ReadVariableOp_1^expconv/BiasAdd/ReadVariableOp^expconv/Conv2D/ReadVariableOp^projbatch/AssignNewValue^projbatch/AssignNewValue_1*^projbatch/FusedBatchNormV3/ReadVariableOp,^projbatch/FusedBatchNormV3/ReadVariableOp_1^projbatch/ReadVariableOp^projbatch/ReadVariableOp_1 ^projconv/BiasAdd/ReadVariableOp^projconv/Conv2D/ReadVariableOp^shortbatch/AssignNewValue^shortbatch/AssignNewValue_1+^shortbatch/FusedBatchNormV3/ReadVariableOp-^shortbatch/FusedBatchNormV3/ReadVariableOp_1^shortbatch/ReadVariableOp^shortbatch/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:џџџџџџџџџ88: : : : : : : : : : : : : : : : : : : : : : 26
depthbatch/AssignNewValuedepthbatch/AssignNewValue2:
depthbatch/AssignNewValue_1depthbatch/AssignNewValue_12X
*depthbatch/FusedBatchNormV3/ReadVariableOp*depthbatch/FusedBatchNormV3/ReadVariableOp2\
,depthbatch/FusedBatchNormV3/ReadVariableOp_1,depthbatch/FusedBatchNormV3/ReadVariableOp_126
depthbatch/ReadVariableOpdepthbatch/ReadVariableOp2:
depthbatch/ReadVariableOp_1depthbatch/ReadVariableOp_12D
 depthconv/BiasAdd/ReadVariableOp depthconv/BiasAdd/ReadVariableOp2H
"depthconv/depthwise/ReadVariableOp"depthconv/depthwise/ReadVariableOp22
expbatch/AssignNewValueexpbatch/AssignNewValue26
expbatch/AssignNewValue_1expbatch/AssignNewValue_12T
(expbatch/FusedBatchNormV3/ReadVariableOp(expbatch/FusedBatchNormV3/ReadVariableOp2X
*expbatch/FusedBatchNormV3/ReadVariableOp_1*expbatch/FusedBatchNormV3/ReadVariableOp_122
expbatch/ReadVariableOpexpbatch/ReadVariableOp26
expbatch/ReadVariableOp_1expbatch/ReadVariableOp_12@
expconv/BiasAdd/ReadVariableOpexpconv/BiasAdd/ReadVariableOp2>
expconv/Conv2D/ReadVariableOpexpconv/Conv2D/ReadVariableOp24
projbatch/AssignNewValueprojbatch/AssignNewValue28
projbatch/AssignNewValue_1projbatch/AssignNewValue_12V
)projbatch/FusedBatchNormV3/ReadVariableOp)projbatch/FusedBatchNormV3/ReadVariableOp2Z
+projbatch/FusedBatchNormV3/ReadVariableOp_1+projbatch/FusedBatchNormV3/ReadVariableOp_124
projbatch/ReadVariableOpprojbatch/ReadVariableOp28
projbatch/ReadVariableOp_1projbatch/ReadVariableOp_12B
projconv/BiasAdd/ReadVariableOpprojconv/BiasAdd/ReadVariableOp2@
projconv/Conv2D/ReadVariableOpprojconv/Conv2D/ReadVariableOp26
shortbatch/AssignNewValueshortbatch/AssignNewValue2:
shortbatch/AssignNewValue_1shortbatch/AssignNewValue_12X
*shortbatch/FusedBatchNormV3/ReadVariableOp*shortbatch/FusedBatchNormV3/ReadVariableOp2\
,shortbatch/FusedBatchNormV3/ReadVariableOp_1,shortbatch/FusedBatchNormV3/ReadVariableOp_126
shortbatch/ReadVariableOpshortbatch/ReadVariableOp2:
shortbatch/ReadVariableOp_1shortbatch/ReadVariableOp_1:R N
/
_output_shapes
:џџџџџџџџџ88

_user_specified_nameX
ї
Х
*__inference_shortbatch_layer_call_fn_51433

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_shortbatch_layer_call_and_return_conditional_losses_47772
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї
Ф
)__inference_projbatch_layer_call_fn_51110

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_projbatch_layer_call_and_return_conditional_losses_47409
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Р

E__inference_shortbatch_layer_call_and_return_conditional_losses_51451

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њ
Д
E__inference_shortbatch_layer_call_and_return_conditional_losses_47772

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ѕ
В
+__inference_Bottleneck2_layer_call_fn_50737
x!
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
	unknown_3:0
	unknown_4:0#
	unknown_5:0
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:0

unknown_10:0$

unknown_11:0

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identityЂStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Bottleneck2_layer_call_and_return_conditional_losses_48437w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameX
ј
В
C__inference_expbatch_layer_call_and_return_conditional_losses_51035

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї
Ф
)__inference_projbatch_layer_call_fn_51358

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_projbatch_layer_call_and_return_conditional_losses_47677
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
П

D__inference_projbatch_layer_call_and_return_conditional_losses_47409

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Р

E__inference_shortbatch_layer_call_and_return_conditional_losses_51203

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
П

D__inference_projbatch_layer_call_and_return_conditional_losses_47677

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ИO
Н
B__inference_model_2_layer_call_and_return_conditional_losses_48969

inputs)
conv2d_18_48851:
conv2d_18_48853:*
batch_normalization_34_48856:*
batch_normalization_34_48858:*
batch_normalization_34_48860:*
batch_normalization_34_48862:+
bottleneck1_48868:
bottleneck1_48870:
bottleneck1_48872:
bottleneck1_48874:
bottleneck1_48876:
bottleneck1_48878:+
bottleneck1_48880:
bottleneck1_48882:
bottleneck1_48884:
bottleneck1_48886:
bottleneck1_48888:
bottleneck1_48890:+
bottleneck1_48892:
bottleneck1_48894:
bottleneck1_48896:
bottleneck1_48898:
bottleneck1_48900:
bottleneck1_48902:
bottleneck1_48904:
bottleneck1_48906:
bottleneck1_48908:
bottleneck1_48910:+
bottleneck2_48915:0
bottleneck2_48917:0
bottleneck2_48919:0
bottleneck2_48921:0
bottleneck2_48923:0
bottleneck2_48925:0+
bottleneck2_48927:0
bottleneck2_48929:0
bottleneck2_48931:0
bottleneck2_48933:0
bottleneck2_48935:0
bottleneck2_48937:0+
bottleneck2_48939:0
bottleneck2_48941:
bottleneck2_48943:
bottleneck2_48945:
bottleneck2_48947:
bottleneck2_48949:
bottleneck2_48951:
bottleneck2_48953:
bottleneck2_48955:
bottleneck2_48957: 
dense_2_48963:		
dense_2_48965:
identityЂ#Bottleneck1/StatefulPartitionedCallЂ#Bottleneck2/StatefulPartitionedCallЂ.batch_normalization_34/StatefulPartitionedCallЂ!conv2d_18/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂ!dropout_6/StatefulPartitionedCallЂ!dropout_7/StatefulPartitionedCallЂ!dropout_8/StatefulPartitionedCallf
tf.identity_2/IdentityIdentityinputs*
T0*1
_output_shapes
:џџџџџџџџџррр
rescaling_2/PartitionedCallPartitionedCalltf.identity_2/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџрр* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_rescaling_2_layer_call_and_return_conditional_losses_47811
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall$rescaling_2/PartitionedCall:output:0conv2d_18_48851conv2d_18_48853*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџpp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_47823
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0batch_normalization_34_48856batch_normalization_34_48858batch_normalization_34_48860batch_normalization_34_48862*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџpp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_47236ј
activation_2/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџpp* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_47843ь
max_pooling2d_6/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ88* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_47256ѓ
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ88* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_48710С
#Bottleneck1/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0bottleneck1_48868bottleneck1_48870bottleneck1_48872bottleneck1_48874bottleneck1_48876bottleneck1_48878bottleneck1_48880bottleneck1_48882bottleneck1_48884bottleneck1_48886bottleneck1_48888bottleneck1_48890bottleneck1_48892bottleneck1_48894bottleneck1_48896bottleneck1_48898bottleneck1_48900bottleneck1_48902bottleneck1_48904bottleneck1_48906bottleneck1_48908bottleneck1_48910*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ88*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Bottleneck1_layer_call_and_return_conditional_losses_48643ѓ
max_pooling2d_7/PartitionedCallPartitionedCall,Bottleneck1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_47524
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_48504С
#Bottleneck2/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0bottleneck2_48915bottleneck2_48917bottleneck2_48919bottleneck2_48921bottleneck2_48923bottleneck2_48925bottleneck2_48927bottleneck2_48929bottleneck2_48931bottleneck2_48933bottleneck2_48935bottleneck2_48937bottleneck2_48939bottleneck2_48941bottleneck2_48943bottleneck2_48945bottleneck2_48947bottleneck2_48949bottleneck2_48951bottleneck2_48953bottleneck2_48955bottleneck2_48957*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Bottleneck2_layer_call_and_return_conditional_losses_48437ѓ
max_pooling2d_8/PartitionedCallPartitionedCall,Bottleneck2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_47792
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_48298о
flatten_2/PartitionedCallPartitionedCall*dropout_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_48135
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_2_48963dense_2_48965*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_48148w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџѕ
NoOpNoOp$^Bottleneck1/StatefulPartitionedCall$^Bottleneck2/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџрр: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#Bottleneck1/StatefulPartitionedCall#Bottleneck1/StatefulPartitionedCall2J
#Bottleneck2/StatefulPartitionedCall#Bottleneck2/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџрр
 
_user_specified_nameinputs

b
)__inference_dropout_7_layer_call_fn_50622

inputs
identityЂStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_48504w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
О

C__inference_expbatch_layer_call_and_return_conditional_losses_51265

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
ї
Х
*__inference_shortbatch_layer_call_fn_51185

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_shortbatch_layer_call_and_return_conditional_losses_47504
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_50627

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџc

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_50612

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Я
ѕ
'__inference_model_2_layer_call_fn_48262
input_3!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:$

unknown_27:0

unknown_28:0

unknown_29:0

unknown_30:0

unknown_31:0

unknown_32:0$

unknown_33:0

unknown_34:0

unknown_35:0

unknown_36:0

unknown_37:0

unknown_38:0$

unknown_39:0

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:		

unknown_50:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_48155o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџрр: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџрр
!
_user_specified_name	input_3
K
в
B__inference_model_2_layer_call_and_return_conditional_losses_49308
input_3)
conv2d_18_49190:
conv2d_18_49192:*
batch_normalization_34_49195:*
batch_normalization_34_49197:*
batch_normalization_34_49199:*
batch_normalization_34_49201:+
bottleneck1_49207:
bottleneck1_49209:
bottleneck1_49211:
bottleneck1_49213:
bottleneck1_49215:
bottleneck1_49217:+
bottleneck1_49219:
bottleneck1_49221:
bottleneck1_49223:
bottleneck1_49225:
bottleneck1_49227:
bottleneck1_49229:+
bottleneck1_49231:
bottleneck1_49233:
bottleneck1_49235:
bottleneck1_49237:
bottleneck1_49239:
bottleneck1_49241:
bottleneck1_49243:
bottleneck1_49245:
bottleneck1_49247:
bottleneck1_49249:+
bottleneck2_49254:0
bottleneck2_49256:0
bottleneck2_49258:0
bottleneck2_49260:0
bottleneck2_49262:0
bottleneck2_49264:0+
bottleneck2_49266:0
bottleneck2_49268:0
bottleneck2_49270:0
bottleneck2_49272:0
bottleneck2_49274:0
bottleneck2_49276:0+
bottleneck2_49278:0
bottleneck2_49280:
bottleneck2_49282:
bottleneck2_49284:
bottleneck2_49286:
bottleneck2_49288:
bottleneck2_49290:
bottleneck2_49292:
bottleneck2_49294:
bottleneck2_49296: 
dense_2_49302:		
dense_2_49304:
identityЂ#Bottleneck1/StatefulPartitionedCallЂ#Bottleneck2/StatefulPartitionedCallЂ.batch_normalization_34/StatefulPartitionedCallЂ!conv2d_18/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallg
tf.identity_2/IdentityIdentityinput_3*
T0*1
_output_shapes
:џџџџџџџџџррр
rescaling_2/PartitionedCallPartitionedCalltf.identity_2/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџрр* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_rescaling_2_layer_call_and_return_conditional_losses_47811
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall$rescaling_2/PartitionedCall:output:0conv2d_18_49190conv2d_18_49192*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџpp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_47823
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0batch_normalization_34_49195batch_normalization_34_49197batch_normalization_34_49199batch_normalization_34_49201*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџpp*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_47205ј
activation_2/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџpp* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_47843ь
max_pooling2d_6/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ88* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_47256у
dropout_6/PartitionedCallPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ88* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_47851С
#Bottleneck1/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0bottleneck1_49207bottleneck1_49209bottleneck1_49211bottleneck1_49213bottleneck1_49215bottleneck1_49217bottleneck1_49219bottleneck1_49221bottleneck1_49223bottleneck1_49225bottleneck1_49227bottleneck1_49229bottleneck1_49231bottleneck1_49233bottleneck1_49235bottleneck1_49237bottleneck1_49239bottleneck1_49241bottleneck1_49243bottleneck1_49245bottleneck1_49247bottleneck1_49249*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ88*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Bottleneck1_layer_call_and_return_conditional_losses_47937ѓ
max_pooling2d_7/PartitionedCallPartitionedCall,Bottleneck1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_47524у
dropout_7/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_47989С
#Bottleneck2/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0bottleneck2_49254bottleneck2_49256bottleneck2_49258bottleneck2_49260bottleneck2_49262bottleneck2_49264bottleneck2_49266bottleneck2_49268bottleneck2_49270bottleneck2_49272bottleneck2_49274bottleneck2_49276bottleneck2_49278bottleneck2_49280bottleneck2_49282bottleneck2_49284bottleneck2_49286bottleneck2_49288bottleneck2_49290bottleneck2_49292bottleneck2_49294bottleneck2_49296*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Bottleneck2_layer_call_and_return_conditional_losses_48075ѓ
max_pooling2d_8/PartitionedCallPartitionedCall,Bottleneck2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_47792у
dropout_8/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_48127ж
flatten_2/PartitionedCallPartitionedCall"dropout_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_48135
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_2_49302dense_2_49304*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_48148w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp$^Bottleneck1/StatefulPartitionedCall$^Bottleneck2/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџрр: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#Bottleneck1/StatefulPartitionedCall#Bottleneck1/StatefulPartitionedCall2J
#Bottleneck2/StatefulPartitionedCall#Bottleneck2/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџрр
!
_user_specified_name	input_3
ф
 [
!__inference__traced_restore_52282
file_prefix;
!assignvariableop_conv2d_18_kernel:/
!assignvariableop_1_conv2d_18_bias:=
/assignvariableop_2_batch_normalization_34_gamma:<
.assignvariableop_3_batch_normalization_34_beta:C
5assignvariableop_4_batch_normalization_34_moving_mean:G
9assignvariableop_5_batch_normalization_34_moving_variance:4
!assignvariableop_6_dense_2_kernel:		-
assignvariableop_7_dense_2_bias:G
-assignvariableop_8_bottleneck1_expconv_kernel:9
+assignvariableop_9_bottleneck1_expconv_bias:<
.assignvariableop_10_bottleneck1_expbatch_gamma:;
-assignvariableop_11_bottleneck1_expbatch_beta:T
:assignvariableop_12_bottleneck1_depthconv_depthwise_kernel:<
.assignvariableop_13_bottleneck1_depthconv_bias:>
0assignvariableop_14_bottleneck1_depthbatch_gamma:=
/assignvariableop_15_bottleneck1_depthbatch_beta:I
/assignvariableop_16_bottleneck1_projconv_kernel:;
-assignvariableop_17_bottleneck1_projconv_bias:=
/assignvariableop_18_bottleneck1_projbatch_gamma:<
.assignvariableop_19_bottleneck1_projbatch_beta:>
0assignvariableop_20_bottleneck1_shortbatch_gamma:=
/assignvariableop_21_bottleneck1_shortbatch_beta:B
4assignvariableop_22_bottleneck1_expbatch_moving_mean:F
8assignvariableop_23_bottleneck1_expbatch_moving_variance:D
6assignvariableop_24_bottleneck1_depthbatch_moving_mean:H
:assignvariableop_25_bottleneck1_depthbatch_moving_variance:C
5assignvariableop_26_bottleneck1_projbatch_moving_mean:G
9assignvariableop_27_bottleneck1_projbatch_moving_variance:D
6assignvariableop_28_bottleneck1_shortbatch_moving_mean:H
:assignvariableop_29_bottleneck1_shortbatch_moving_variance:H
.assignvariableop_30_bottleneck2_expconv_kernel:0:
,assignvariableop_31_bottleneck2_expconv_bias:0<
.assignvariableop_32_bottleneck2_expbatch_gamma:0;
-assignvariableop_33_bottleneck2_expbatch_beta:0T
:assignvariableop_34_bottleneck2_depthconv_depthwise_kernel:0<
.assignvariableop_35_bottleneck2_depthconv_bias:0>
0assignvariableop_36_bottleneck2_depthbatch_gamma:0=
/assignvariableop_37_bottleneck2_depthbatch_beta:0I
/assignvariableop_38_bottleneck2_projconv_kernel:0;
-assignvariableop_39_bottleneck2_projconv_bias:=
/assignvariableop_40_bottleneck2_projbatch_gamma:<
.assignvariableop_41_bottleneck2_projbatch_beta:>
0assignvariableop_42_bottleneck2_shortbatch_gamma:=
/assignvariableop_43_bottleneck2_shortbatch_beta:B
4assignvariableop_44_bottleneck2_expbatch_moving_mean:0F
8assignvariableop_45_bottleneck2_expbatch_moving_variance:0D
6assignvariableop_46_bottleneck2_depthbatch_moving_mean:0H
:assignvariableop_47_bottleneck2_depthbatch_moving_variance:0C
5assignvariableop_48_bottleneck2_projbatch_moving_mean:G
9assignvariableop_49_bottleneck2_projbatch_moving_variance:D
6assignvariableop_50_bottleneck2_shortbatch_moving_mean:H
:assignvariableop_51_bottleneck2_shortbatch_moving_variance:'
assignvariableop_52_adam_iter:	 )
assignvariableop_53_adam_beta_1: )
assignvariableop_54_adam_beta_2: (
assignvariableop_55_adam_decay: %
assignvariableop_56_total_2: %
assignvariableop_57_count_2: %
assignvariableop_58_total_1: %
assignvariableop_59_count_1: #
assignvariableop_60_total: #
assignvariableop_61_count: E
+assignvariableop_62_adam_conv2d_18_kernel_m:7
)assignvariableop_63_adam_conv2d_18_bias_m:E
7assignvariableop_64_adam_batch_normalization_34_gamma_m:D
6assignvariableop_65_adam_batch_normalization_34_beta_m:<
)assignvariableop_66_adam_dense_2_kernel_m:		5
'assignvariableop_67_adam_dense_2_bias_m:O
5assignvariableop_68_adam_bottleneck1_expconv_kernel_m:A
3assignvariableop_69_adam_bottleneck1_expconv_bias_m:C
5assignvariableop_70_adam_bottleneck1_expbatch_gamma_m:B
4assignvariableop_71_adam_bottleneck1_expbatch_beta_m:[
Aassignvariableop_72_adam_bottleneck1_depthconv_depthwise_kernel_m:C
5assignvariableop_73_adam_bottleneck1_depthconv_bias_m:E
7assignvariableop_74_adam_bottleneck1_depthbatch_gamma_m:D
6assignvariableop_75_adam_bottleneck1_depthbatch_beta_m:P
6assignvariableop_76_adam_bottleneck1_projconv_kernel_m:B
4assignvariableop_77_adam_bottleneck1_projconv_bias_m:D
6assignvariableop_78_adam_bottleneck1_projbatch_gamma_m:C
5assignvariableop_79_adam_bottleneck1_projbatch_beta_m:E
7assignvariableop_80_adam_bottleneck1_shortbatch_gamma_m:D
6assignvariableop_81_adam_bottleneck1_shortbatch_beta_m:O
5assignvariableop_82_adam_bottleneck2_expconv_kernel_m:0A
3assignvariableop_83_adam_bottleneck2_expconv_bias_m:0C
5assignvariableop_84_adam_bottleneck2_expbatch_gamma_m:0B
4assignvariableop_85_adam_bottleneck2_expbatch_beta_m:0[
Aassignvariableop_86_adam_bottleneck2_depthconv_depthwise_kernel_m:0C
5assignvariableop_87_adam_bottleneck2_depthconv_bias_m:0E
7assignvariableop_88_adam_bottleneck2_depthbatch_gamma_m:0D
6assignvariableop_89_adam_bottleneck2_depthbatch_beta_m:0P
6assignvariableop_90_adam_bottleneck2_projconv_kernel_m:0B
4assignvariableop_91_adam_bottleneck2_projconv_bias_m:D
6assignvariableop_92_adam_bottleneck2_projbatch_gamma_m:C
5assignvariableop_93_adam_bottleneck2_projbatch_beta_m:E
7assignvariableop_94_adam_bottleneck2_shortbatch_gamma_m:D
6assignvariableop_95_adam_bottleneck2_shortbatch_beta_m:E
+assignvariableop_96_adam_conv2d_18_kernel_v:7
)assignvariableop_97_adam_conv2d_18_bias_v:E
7assignvariableop_98_adam_batch_normalization_34_gamma_v:D
6assignvariableop_99_adam_batch_normalization_34_beta_v:=
*assignvariableop_100_adam_dense_2_kernel_v:		6
(assignvariableop_101_adam_dense_2_bias_v:P
6assignvariableop_102_adam_bottleneck1_expconv_kernel_v:B
4assignvariableop_103_adam_bottleneck1_expconv_bias_v:D
6assignvariableop_104_adam_bottleneck1_expbatch_gamma_v:C
5assignvariableop_105_adam_bottleneck1_expbatch_beta_v:\
Bassignvariableop_106_adam_bottleneck1_depthconv_depthwise_kernel_v:D
6assignvariableop_107_adam_bottleneck1_depthconv_bias_v:F
8assignvariableop_108_adam_bottleneck1_depthbatch_gamma_v:E
7assignvariableop_109_adam_bottleneck1_depthbatch_beta_v:Q
7assignvariableop_110_adam_bottleneck1_projconv_kernel_v:C
5assignvariableop_111_adam_bottleneck1_projconv_bias_v:E
7assignvariableop_112_adam_bottleneck1_projbatch_gamma_v:D
6assignvariableop_113_adam_bottleneck1_projbatch_beta_v:F
8assignvariableop_114_adam_bottleneck1_shortbatch_gamma_v:E
7assignvariableop_115_adam_bottleneck1_shortbatch_beta_v:P
6assignvariableop_116_adam_bottleneck2_expconv_kernel_v:0B
4assignvariableop_117_adam_bottleneck2_expconv_bias_v:0D
6assignvariableop_118_adam_bottleneck2_expbatch_gamma_v:0C
5assignvariableop_119_adam_bottleneck2_expbatch_beta_v:0\
Bassignvariableop_120_adam_bottleneck2_depthconv_depthwise_kernel_v:0D
6assignvariableop_121_adam_bottleneck2_depthconv_bias_v:0F
8assignvariableop_122_adam_bottleneck2_depthbatch_gamma_v:0E
7assignvariableop_123_adam_bottleneck2_depthbatch_beta_v:0Q
7assignvariableop_124_adam_bottleneck2_projconv_kernel_v:0C
5assignvariableop_125_adam_bottleneck2_projconv_bias_v:E
7assignvariableop_126_adam_bottleneck2_projbatch_gamma_v:D
6assignvariableop_127_adam_bottleneck2_projbatch_beta_v:F
8assignvariableop_128_adam_bottleneck2_shortbatch_gamma_v:E
7assignvariableop_129_adam_bottleneck2_shortbatch_beta_v:
identity_131ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_100ЂAssignVariableOp_101ЂAssignVariableOp_102ЂAssignVariableOp_103ЂAssignVariableOp_104ЂAssignVariableOp_105ЂAssignVariableOp_106ЂAssignVariableOp_107ЂAssignVariableOp_108ЂAssignVariableOp_109ЂAssignVariableOp_11ЂAssignVariableOp_110ЂAssignVariableOp_111ЂAssignVariableOp_112ЂAssignVariableOp_113ЂAssignVariableOp_114ЂAssignVariableOp_115ЂAssignVariableOp_116ЂAssignVariableOp_117ЂAssignVariableOp_118ЂAssignVariableOp_119ЂAssignVariableOp_12ЂAssignVariableOp_120ЂAssignVariableOp_121ЂAssignVariableOp_122ЂAssignVariableOp_123ЂAssignVariableOp_124ЂAssignVariableOp_125ЂAssignVariableOp_126ЂAssignVariableOp_127ЂAssignVariableOp_128ЂAssignVariableOp_129ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_70ЂAssignVariableOp_71ЂAssignVariableOp_72ЂAssignVariableOp_73ЂAssignVariableOp_74ЂAssignVariableOp_75ЂAssignVariableOp_76ЂAssignVariableOp_77ЂAssignVariableOp_78ЂAssignVariableOp_79ЂAssignVariableOp_8ЂAssignVariableOp_80ЂAssignVariableOp_81ЂAssignVariableOp_82ЂAssignVariableOp_83ЂAssignVariableOp_84ЂAssignVariableOp_85ЂAssignVariableOp_86ЂAssignVariableOp_87ЂAssignVariableOp_88ЂAssignVariableOp_89ЂAssignVariableOp_9ЂAssignVariableOp_90ЂAssignVariableOp_91ЂAssignVariableOp_92ЂAssignVariableOp_93ЂAssignVariableOp_94ЂAssignVariableOp_95ЂAssignVariableOp_96ЂAssignVariableOp_97ЂAssignVariableOp_98ЂAssignVariableOp_99Х<
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*ъ;
valueр;Bн;B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHћ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*
valueBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Д
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ђ
_output_shapes
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_18_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_18_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_34_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_34_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_34_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_34_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp-assignvariableop_8_bottleneck1_expconv_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp+assignvariableop_9_bottleneck1_expconv_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp.assignvariableop_10_bottleneck1_expbatch_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp-assignvariableop_11_bottleneck1_expbatch_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_12AssignVariableOp:assignvariableop_12_bottleneck1_depthconv_depthwise_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp.assignvariableop_13_bottleneck1_depthconv_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_14AssignVariableOp0assignvariableop_14_bottleneck1_depthbatch_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_15AssignVariableOp/assignvariableop_15_bottleneck1_depthbatch_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_16AssignVariableOp/assignvariableop_16_bottleneck1_projconv_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp-assignvariableop_17_bottleneck1_projconv_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_18AssignVariableOp/assignvariableop_18_bottleneck1_projbatch_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp.assignvariableop_19_bottleneck1_projbatch_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_20AssignVariableOp0assignvariableop_20_bottleneck1_shortbatch_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_21AssignVariableOp/assignvariableop_21_bottleneck1_shortbatch_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_22AssignVariableOp4assignvariableop_22_bottleneck1_expbatch_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_23AssignVariableOp8assignvariableop_23_bottleneck1_expbatch_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_24AssignVariableOp6assignvariableop_24_bottleneck1_depthbatch_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_25AssignVariableOp:assignvariableop_25_bottleneck1_depthbatch_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_26AssignVariableOp5assignvariableop_26_bottleneck1_projbatch_moving_meanIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_27AssignVariableOp9assignvariableop_27_bottleneck1_projbatch_moving_varianceIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_28AssignVariableOp6assignvariableop_28_bottleneck1_shortbatch_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_29AssignVariableOp:assignvariableop_29_bottleneck1_shortbatch_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp.assignvariableop_30_bottleneck2_expconv_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp,assignvariableop_31_bottleneck2_expconv_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp.assignvariableop_32_bottleneck2_expbatch_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp-assignvariableop_33_bottleneck2_expbatch_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_34AssignVariableOp:assignvariableop_34_bottleneck2_depthconv_depthwise_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp.assignvariableop_35_bottleneck2_depthconv_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_36AssignVariableOp0assignvariableop_36_bottleneck2_depthbatch_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_37AssignVariableOp/assignvariableop_37_bottleneck2_depthbatch_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_38AssignVariableOp/assignvariableop_38_bottleneck2_projconv_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp-assignvariableop_39_bottleneck2_projconv_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_40AssignVariableOp/assignvariableop_40_bottleneck2_projbatch_gammaIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp.assignvariableop_41_bottleneck2_projbatch_betaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_42AssignVariableOp0assignvariableop_42_bottleneck2_shortbatch_gammaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_43AssignVariableOp/assignvariableop_43_bottleneck2_shortbatch_betaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_44AssignVariableOp4assignvariableop_44_bottleneck2_expbatch_moving_meanIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_45AssignVariableOp8assignvariableop_45_bottleneck2_expbatch_moving_varianceIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_46AssignVariableOp6assignvariableop_46_bottleneck2_depthbatch_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_47AssignVariableOp:assignvariableop_47_bottleneck2_depthbatch_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_48AssignVariableOp5assignvariableop_48_bottleneck2_projbatch_moving_meanIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_49AssignVariableOp9assignvariableop_49_bottleneck2_projbatch_moving_varianceIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_50AssignVariableOp6assignvariableop_50_bottleneck2_shortbatch_moving_meanIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_51AssignVariableOp:assignvariableop_51_bottleneck2_shortbatch_moving_varianceIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_52AssignVariableOpassignvariableop_52_adam_iterIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOpassignvariableop_53_adam_beta_1Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOpassignvariableop_54_adam_beta_2Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOpassignvariableop_55_adam_decayIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOpassignvariableop_56_total_2Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOpassignvariableop_57_count_2Identity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOpassignvariableop_58_total_1Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_59AssignVariableOpassignvariableop_59_count_1Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_60AssignVariableOpassignvariableop_60_totalIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOpassignvariableop_61_countIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOp+assignvariableop_62_adam_conv2d_18_kernel_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOp)assignvariableop_63_adam_conv2d_18_bias_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_64AssignVariableOp7assignvariableop_64_adam_batch_normalization_34_gamma_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_batch_normalization_34_beta_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_dense_2_kernel_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_67AssignVariableOp'assignvariableop_67_adam_dense_2_bias_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_68AssignVariableOp5assignvariableop_68_adam_bottleneck1_expconv_kernel_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_69AssignVariableOp3assignvariableop_69_adam_bottleneck1_expconv_bias_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adam_bottleneck1_expbatch_gamma_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_71AssignVariableOp4assignvariableop_71_adam_bottleneck1_expbatch_beta_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_72AssignVariableOpAassignvariableop_72_adam_bottleneck1_depthconv_depthwise_kernel_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_73AssignVariableOp5assignvariableop_73_adam_bottleneck1_depthconv_bias_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_74AssignVariableOp7assignvariableop_74_adam_bottleneck1_depthbatch_gamma_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_75AssignVariableOp6assignvariableop_75_adam_bottleneck1_depthbatch_beta_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_76AssignVariableOp6assignvariableop_76_adam_bottleneck1_projconv_kernel_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_77AssignVariableOp4assignvariableop_77_adam_bottleneck1_projconv_bias_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_78AssignVariableOp6assignvariableop_78_adam_bottleneck1_projbatch_gamma_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_79AssignVariableOp5assignvariableop_79_adam_bottleneck1_projbatch_beta_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_80AssignVariableOp7assignvariableop_80_adam_bottleneck1_shortbatch_gamma_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_81AssignVariableOp6assignvariableop_81_adam_bottleneck1_shortbatch_beta_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_82AssignVariableOp5assignvariableop_82_adam_bottleneck2_expconv_kernel_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_83AssignVariableOp3assignvariableop_83_adam_bottleneck2_expconv_bias_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_84AssignVariableOp5assignvariableop_84_adam_bottleneck2_expbatch_gamma_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_85AssignVariableOp4assignvariableop_85_adam_bottleneck2_expbatch_beta_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_86AssignVariableOpAassignvariableop_86_adam_bottleneck2_depthconv_depthwise_kernel_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_87AssignVariableOp5assignvariableop_87_adam_bottleneck2_depthconv_bias_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_88AssignVariableOp7assignvariableop_88_adam_bottleneck2_depthbatch_gamma_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_89AssignVariableOp6assignvariableop_89_adam_bottleneck2_depthbatch_beta_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_90AssignVariableOp6assignvariableop_90_adam_bottleneck2_projconv_kernel_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_91AssignVariableOp4assignvariableop_91_adam_bottleneck2_projconv_bias_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_92AssignVariableOp6assignvariableop_92_adam_bottleneck2_projbatch_gamma_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_93AssignVariableOp5assignvariableop_93_adam_bottleneck2_projbatch_beta_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_94AssignVariableOp7assignvariableop_94_adam_bottleneck2_shortbatch_gamma_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_95AssignVariableOp6assignvariableop_95_adam_bottleneck2_shortbatch_beta_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_96AssignVariableOp+assignvariableop_96_adam_conv2d_18_kernel_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_97AssignVariableOp)assignvariableop_97_adam_conv2d_18_bias_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_98AssignVariableOp7assignvariableop_98_adam_batch_normalization_34_gamma_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_99AssignVariableOp6assignvariableop_99_adam_batch_normalization_34_beta_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_100AssignVariableOp*assignvariableop_100_adam_dense_2_kernel_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_101AssignVariableOp(assignvariableop_101_adam_dense_2_bias_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_102AssignVariableOp6assignvariableop_102_adam_bottleneck1_expconv_kernel_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_103AssignVariableOp4assignvariableop_103_adam_bottleneck1_expconv_bias_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_104AssignVariableOp6assignvariableop_104_adam_bottleneck1_expbatch_gamma_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_105AssignVariableOp5assignvariableop_105_adam_bottleneck1_expbatch_beta_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_106AssignVariableOpBassignvariableop_106_adam_bottleneck1_depthconv_depthwise_kernel_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_107AssignVariableOp6assignvariableop_107_adam_bottleneck1_depthconv_bias_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_108AssignVariableOp8assignvariableop_108_adam_bottleneck1_depthbatch_gamma_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_109AssignVariableOp7assignvariableop_109_adam_bottleneck1_depthbatch_beta_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_110AssignVariableOp7assignvariableop_110_adam_bottleneck1_projconv_kernel_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_111AssignVariableOp5assignvariableop_111_adam_bottleneck1_projconv_bias_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_112AssignVariableOp7assignvariableop_112_adam_bottleneck1_projbatch_gamma_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_113AssignVariableOp6assignvariableop_113_adam_bottleneck1_projbatch_beta_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_114AssignVariableOp8assignvariableop_114_adam_bottleneck1_shortbatch_gamma_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_115AssignVariableOp7assignvariableop_115_adam_bottleneck1_shortbatch_beta_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_116AssignVariableOp6assignvariableop_116_adam_bottleneck2_expconv_kernel_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_117AssignVariableOp4assignvariableop_117_adam_bottleneck2_expconv_bias_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_118AssignVariableOp6assignvariableop_118_adam_bottleneck2_expbatch_gamma_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_119AssignVariableOp5assignvariableop_119_adam_bottleneck2_expbatch_beta_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_120AssignVariableOpBassignvariableop_120_adam_bottleneck2_depthconv_depthwise_kernel_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_121AssignVariableOp6assignvariableop_121_adam_bottleneck2_depthconv_bias_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_122AssignVariableOp8assignvariableop_122_adam_bottleneck2_depthbatch_gamma_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_123AssignVariableOp7assignvariableop_123_adam_bottleneck2_depthbatch_beta_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_124AssignVariableOp7assignvariableop_124_adam_bottleneck2_projconv_kernel_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_125AssignVariableOp5assignvariableop_125_adam_bottleneck2_projconv_bias_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_126AssignVariableOp7assignvariableop_126_adam_bottleneck2_projbatch_gamma_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_127AssignVariableOp6assignvariableop_127_adam_bottleneck2_projbatch_beta_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_128AssignVariableOp8assignvariableop_128_adam_bottleneck2_shortbatch_gamma_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_129AssignVariableOp7assignvariableop_129_adam_bottleneck2_shortbatch_beta_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_130Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_131IdentityIdentity_130:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_131Identity_131:output:0*
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_47792

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ј
b
F__inference_rescaling_2_layer_call_and_return_conditional_losses_47811

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџррd
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџррY
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:џџџџџџџџџрр"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџрр:Y U
1
_output_shapes
:џџџџџџџџџрр
 
_user_specified_nameinputs
Џ
E
)__inference_flatten_2_layer_call_fn_50947

inputs
identityА
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_48135a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_50915

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ь

Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_47205

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
­
В
+__inference_Bottleneck2_layer_call_fn_50688
x!
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
	unknown_3:0
	unknown_4:0#
	unknown_5:0
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:0

unknown_10:0$

unknown_11:0

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Bottleneck2_layer_call_and_return_conditional_losses_48075w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameX
ѕ
У
(__inference_expbatch_layer_call_fn_50986

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_expbatch_layer_call_and_return_conditional_losses_47281
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Љ
ё
#__inference_signature_wrapper_49546
input_3!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:$

unknown_27:0

unknown_28:0

unknown_29:0

unknown_30:0

unknown_31:0

unknown_32:0$

unknown_33:0

unknown_34:0

unknown_35:0

unknown_36:0

unknown_37:0

unknown_38:0$

unknown_39:0

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:		

unknown_50:
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_47183o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџрр: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџрр
!
_user_specified_name	input_3

f
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_50309

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
О

c
D__inference_dropout_7_layer_call_and_return_conditional_losses_50639

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџq
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџa
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
О

c
D__inference_dropout_8_layer_call_and_return_conditional_losses_50942

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџq
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџa
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ц
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_48135

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѕ
Ф
)__inference_projbatch_layer_call_fn_51371

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_projbatch_layer_call_and_return_conditional_losses_47708
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
О

c
D__inference_dropout_7_layer_call_and_return_conditional_losses_48504

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџq
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџa
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЛO
О
B__inference_model_2_layer_call_and_return_conditional_losses_49431
input_3)
conv2d_18_49313:
conv2d_18_49315:*
batch_normalization_34_49318:*
batch_normalization_34_49320:*
batch_normalization_34_49322:*
batch_normalization_34_49324:+
bottleneck1_49330:
bottleneck1_49332:
bottleneck1_49334:
bottleneck1_49336:
bottleneck1_49338:
bottleneck1_49340:+
bottleneck1_49342:
bottleneck1_49344:
bottleneck1_49346:
bottleneck1_49348:
bottleneck1_49350:
bottleneck1_49352:+
bottleneck1_49354:
bottleneck1_49356:
bottleneck1_49358:
bottleneck1_49360:
bottleneck1_49362:
bottleneck1_49364:
bottleneck1_49366:
bottleneck1_49368:
bottleneck1_49370:
bottleneck1_49372:+
bottleneck2_49377:0
bottleneck2_49379:0
bottleneck2_49381:0
bottleneck2_49383:0
bottleneck2_49385:0
bottleneck2_49387:0+
bottleneck2_49389:0
bottleneck2_49391:0
bottleneck2_49393:0
bottleneck2_49395:0
bottleneck2_49397:0
bottleneck2_49399:0+
bottleneck2_49401:0
bottleneck2_49403:
bottleneck2_49405:
bottleneck2_49407:
bottleneck2_49409:
bottleneck2_49411:
bottleneck2_49413:
bottleneck2_49415:
bottleneck2_49417:
bottleneck2_49419: 
dense_2_49425:		
dense_2_49427:
identityЂ#Bottleneck1/StatefulPartitionedCallЂ#Bottleneck2/StatefulPartitionedCallЂ.batch_normalization_34/StatefulPartitionedCallЂ!conv2d_18/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂ!dropout_6/StatefulPartitionedCallЂ!dropout_7/StatefulPartitionedCallЂ!dropout_8/StatefulPartitionedCallg
tf.identity_2/IdentityIdentityinput_3*
T0*1
_output_shapes
:џџџџџџџџџррр
rescaling_2/PartitionedCallPartitionedCalltf.identity_2/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџрр* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_rescaling_2_layer_call_and_return_conditional_losses_47811
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall$rescaling_2/PartitionedCall:output:0conv2d_18_49313conv2d_18_49315*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџpp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_47823
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0batch_normalization_34_49318batch_normalization_34_49320batch_normalization_34_49322batch_normalization_34_49324*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџpp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_47236ј
activation_2/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџpp* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_47843ь
max_pooling2d_6/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ88* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_47256ѓ
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ88* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_48710С
#Bottleneck1/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0bottleneck1_49330bottleneck1_49332bottleneck1_49334bottleneck1_49336bottleneck1_49338bottleneck1_49340bottleneck1_49342bottleneck1_49344bottleneck1_49346bottleneck1_49348bottleneck1_49350bottleneck1_49352bottleneck1_49354bottleneck1_49356bottleneck1_49358bottleneck1_49360bottleneck1_49362bottleneck1_49364bottleneck1_49366bottleneck1_49368bottleneck1_49370bottleneck1_49372*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ88*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Bottleneck1_layer_call_and_return_conditional_losses_48643ѓ
max_pooling2d_7/PartitionedCallPartitionedCall,Bottleneck1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_47524
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_48504С
#Bottleneck2/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0bottleneck2_49377bottleneck2_49379bottleneck2_49381bottleneck2_49383bottleneck2_49385bottleneck2_49387bottleneck2_49389bottleneck2_49391bottleneck2_49393bottleneck2_49395bottleneck2_49397bottleneck2_49399bottleneck2_49401bottleneck2_49403bottleneck2_49405bottleneck2_49407bottleneck2_49409bottleneck2_49411bottleneck2_49413bottleneck2_49415bottleneck2_49417bottleneck2_49419*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Bottleneck2_layer_call_and_return_conditional_losses_48437ѓ
max_pooling2d_8/PartitionedCallPartitionedCall,Bottleneck2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_47792
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_48298о
flatten_2/PartitionedCallPartitionedCall*dropout_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_48135
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_2_49425dense_2_49427*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_48148w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџѕ
NoOpNoOp$^Bottleneck1/StatefulPartitionedCall$^Bottleneck2/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџрр: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#Bottleneck1/StatefulPartitionedCall#Bottleneck1/StatefulPartitionedCall2J
#Bottleneck2/StatefulPartitionedCall#Bottleneck2/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџрр
!
_user_specified_name	input_3
Ѕ
В
+__inference_Bottleneck1_layer_call_fn_50434
x!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identityЂStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ88*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Bottleneck1_layer_call_and_return_conditional_losses_48643w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ88`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:џџџџџџџџџ88: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:џџџџџџџџџ88

_user_specified_nameX
K
б
B__inference_model_2_layer_call_and_return_conditional_losses_48155

inputs)
conv2d_18_47824:
conv2d_18_47826:*
batch_normalization_34_47829:*
batch_normalization_34_47831:*
batch_normalization_34_47833:*
batch_normalization_34_47835:+
bottleneck1_47938:
bottleneck1_47940:
bottleneck1_47942:
bottleneck1_47944:
bottleneck1_47946:
bottleneck1_47948:+
bottleneck1_47950:
bottleneck1_47952:
bottleneck1_47954:
bottleneck1_47956:
bottleneck1_47958:
bottleneck1_47960:+
bottleneck1_47962:
bottleneck1_47964:
bottleneck1_47966:
bottleneck1_47968:
bottleneck1_47970:
bottleneck1_47972:
bottleneck1_47974:
bottleneck1_47976:
bottleneck1_47978:
bottleneck1_47980:+
bottleneck2_48076:0
bottleneck2_48078:0
bottleneck2_48080:0
bottleneck2_48082:0
bottleneck2_48084:0
bottleneck2_48086:0+
bottleneck2_48088:0
bottleneck2_48090:0
bottleneck2_48092:0
bottleneck2_48094:0
bottleneck2_48096:0
bottleneck2_48098:0+
bottleneck2_48100:0
bottleneck2_48102:
bottleneck2_48104:
bottleneck2_48106:
bottleneck2_48108:
bottleneck2_48110:
bottleneck2_48112:
bottleneck2_48114:
bottleneck2_48116:
bottleneck2_48118: 
dense_2_48149:		
dense_2_48151:
identityЂ#Bottleneck1/StatefulPartitionedCallЂ#Bottleneck2/StatefulPartitionedCallЂ.batch_normalization_34/StatefulPartitionedCallЂ!conv2d_18/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallf
tf.identity_2/IdentityIdentityinputs*
T0*1
_output_shapes
:џџџџџџџџџррр
rescaling_2/PartitionedCallPartitionedCalltf.identity_2/Identity:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџрр* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_rescaling_2_layer_call_and_return_conditional_losses_47811
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall$rescaling_2/PartitionedCall:output:0conv2d_18_47824conv2d_18_47826*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџpp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_47823
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0batch_normalization_34_47829batch_normalization_34_47831batch_normalization_34_47833batch_normalization_34_47835*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџpp*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_47205ј
activation_2/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџpp* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_47843ь
max_pooling2d_6/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ88* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_47256у
dropout_6/PartitionedCallPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ88* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_47851С
#Bottleneck1/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0bottleneck1_47938bottleneck1_47940bottleneck1_47942bottleneck1_47944bottleneck1_47946bottleneck1_47948bottleneck1_47950bottleneck1_47952bottleneck1_47954bottleneck1_47956bottleneck1_47958bottleneck1_47960bottleneck1_47962bottleneck1_47964bottleneck1_47966bottleneck1_47968bottleneck1_47970bottleneck1_47972bottleneck1_47974bottleneck1_47976bottleneck1_47978bottleneck1_47980*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ88*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Bottleneck1_layer_call_and_return_conditional_losses_47937ѓ
max_pooling2d_7/PartitionedCallPartitionedCall,Bottleneck1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_47524у
dropout_7/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_47989С
#Bottleneck2/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0bottleneck2_48076bottleneck2_48078bottleneck2_48080bottleneck2_48082bottleneck2_48084bottleneck2_48086bottleneck2_48088bottleneck2_48090bottleneck2_48092bottleneck2_48094bottleneck2_48096bottleneck2_48098bottleneck2_48100bottleneck2_48102bottleneck2_48104bottleneck2_48106bottleneck2_48108bottleneck2_48110bottleneck2_48112bottleneck2_48114bottleneck2_48116bottleneck2_48118*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Bottleneck2_layer_call_and_return_conditional_losses_48075ѓ
max_pooling2d_8/PartitionedCallPartitionedCall,Bottleneck2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_47792у
dropout_8/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_48127ж
flatten_2/PartitionedCallPartitionedCall"dropout_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_48135
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_2_48149dense_2_48151*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_48148w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp$^Bottleneck1/StatefulPartitionedCall$^Bottleneck2/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџрр: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#Bottleneck1/StatefulPartitionedCall#Bottleneck1/StatefulPartitionedCall2J
#Bottleneck2/StatefulPartitionedCall#Bottleneck2/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџрр
 
_user_specified_nameinputs
ї
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_47989

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџc

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ь

Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_50271

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њ
Д
E__inference_shortbatch_layer_call_and_return_conditional_losses_47504

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ

§
D__inference_conv2d_18_layer_call_and_return_conditional_losses_47823

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџpp*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџppg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџрр: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџрр
 
_user_specified_nameinputs
ѕ
Ф
)__inference_projbatch_layer_call_fn_51123

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_projbatch_layer_call_and_return_conditional_losses_47440
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
юX

F__inference_Bottleneck2_layer_call_and_return_conditional_losses_50821
x@
&expconv_conv2d_readvariableop_resource:05
'expconv_biasadd_readvariableop_resource:0.
 expbatch_readvariableop_resource:00
"expbatch_readvariableop_1_resource:0?
1expbatch_fusedbatchnormv3_readvariableop_resource:0A
3expbatch_fusedbatchnormv3_readvariableop_1_resource:0E
+depthconv_depthwise_readvariableop_resource:07
)depthconv_biasadd_readvariableop_resource:00
"depthbatch_readvariableop_resource:02
$depthbatch_readvariableop_1_resource:0A
3depthbatch_fusedbatchnormv3_readvariableop_resource:0C
5depthbatch_fusedbatchnormv3_readvariableop_1_resource:0A
'projconv_conv2d_readvariableop_resource:06
(projconv_biasadd_readvariableop_resource:/
!projbatch_readvariableop_resource:1
#projbatch_readvariableop_1_resource:@
2projbatch_fusedbatchnormv3_readvariableop_resource:B
4projbatch_fusedbatchnormv3_readvariableop_1_resource:0
"shortbatch_readvariableop_resource:2
$shortbatch_readvariableop_1_resource:A
3shortbatch_fusedbatchnormv3_readvariableop_resource:C
5shortbatch_fusedbatchnormv3_readvariableop_1_resource:
identityЂ*depthbatch/FusedBatchNormV3/ReadVariableOpЂ,depthbatch/FusedBatchNormV3/ReadVariableOp_1Ђdepthbatch/ReadVariableOpЂdepthbatch/ReadVariableOp_1Ђ depthconv/BiasAdd/ReadVariableOpЂ"depthconv/depthwise/ReadVariableOpЂ(expbatch/FusedBatchNormV3/ReadVariableOpЂ*expbatch/FusedBatchNormV3/ReadVariableOp_1Ђexpbatch/ReadVariableOpЂexpbatch/ReadVariableOp_1Ђexpconv/BiasAdd/ReadVariableOpЂexpconv/Conv2D/ReadVariableOpЂ)projbatch/FusedBatchNormV3/ReadVariableOpЂ+projbatch/FusedBatchNormV3/ReadVariableOp_1Ђprojbatch/ReadVariableOpЂprojbatch/ReadVariableOp_1Ђprojconv/BiasAdd/ReadVariableOpЂprojconv/Conv2D/ReadVariableOpЂ*shortbatch/FusedBatchNormV3/ReadVariableOpЂ,shortbatch/FusedBatchNormV3/ReadVariableOp_1Ђshortbatch/ReadVariableOpЂshortbatch/ReadVariableOp_1
expconv/Conv2D/ReadVariableOpReadVariableOp&expconv_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0Є
expconv/Conv2DConv2Dx%expconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides

expconv/BiasAdd/ReadVariableOpReadVariableOp'expconv_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
expconv/BiasAddBiasAddexpconv/Conv2D:output:0&expconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0t
expbatch/ReadVariableOpReadVariableOp expbatch_readvariableop_resource*
_output_shapes
:0*
dtype0x
expbatch/ReadVariableOp_1ReadVariableOp"expbatch_readvariableop_1_resource*
_output_shapes
:0*
dtype0
(expbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp1expbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
*expbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3expbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0ѕ
expbatch/FusedBatchNormV3FusedBatchNormV3expconv/BiasAdd:output:0expbatch/ReadVariableOp:value:0!expbatch/ReadVariableOp_1:value:00expbatch/FusedBatchNormV3/ReadVariableOp:value:02expbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( p
activation/ReluReluexpbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ0
"depthconv/depthwise/ReadVariableOpReadVariableOp+depthconv_depthwise_readvariableop_resource*&
_output_shapes
:0*
dtype0r
depthconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      r
!depthconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      й
depthconv/depthwiseDepthwiseConv2dNativeactivation/Relu:activations:0*depthconv/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides

 depthconv/BiasAdd/ReadVariableOpReadVariableOp)depthconv_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
depthconv/BiasAddBiasAdddepthconv/depthwise:output:0(depthconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0x
depthbatch/ReadVariableOpReadVariableOp"depthbatch_readvariableop_resource*
_output_shapes
:0*
dtype0|
depthbatch/ReadVariableOp_1ReadVariableOp$depthbatch_readvariableop_1_resource*
_output_shapes
:0*
dtype0
*depthbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp3depthbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
,depthbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5depthbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0
depthbatch/FusedBatchNormV3FusedBatchNormV3depthconv/BiasAdd:output:0!depthbatch/ReadVariableOp:value:0#depthbatch/ReadVariableOp_1:value:02depthbatch/FusedBatchNormV3/ReadVariableOp:value:04depthbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( t
activation_1/ReluReludepthbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ0
projconv/Conv2D/ReadVariableOpReadVariableOp'projconv_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0Ф
projconv/Conv2DConv2Dactivation_1/Relu:activations:0&projconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

projconv/BiasAdd/ReadVariableOpReadVariableOp(projconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
projconv/BiasAddBiasAddprojconv/Conv2D:output:0'projconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџv
projbatch/ReadVariableOpReadVariableOp!projbatch_readvariableop_resource*
_output_shapes
:*
dtype0z
projbatch/ReadVariableOp_1ReadVariableOp#projbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
)projbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp2projbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
+projbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4projbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ћ
projbatch/FusedBatchNormV3FusedBatchNormV3projconv/BiasAdd:output:0 projbatch/ReadVariableOp:value:0"projbatch/ReadVariableOp_1:value:01projbatch/FusedBatchNormV3/ReadVariableOp:value:03projbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
is_training( x
shortbatch/ReadVariableOpReadVariableOp"shortbatch_readvariableop_resource*
_output_shapes
:*
dtype0|
shortbatch/ReadVariableOp_1ReadVariableOp$shortbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
*shortbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp3shortbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
,shortbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5shortbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ш
shortbatch/FusedBatchNormV3FusedBatchNormV3x!shortbatch/ReadVariableOp:value:0#shortbatch/ReadVariableOp_1:value:02shortbatch/FusedBatchNormV3/ReadVariableOp:value:04shortbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 
add/addAddV2projbatch/FusedBatchNormV3:y:0shortbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ`
activation_2/ReluReluadd/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџv
IdentityIdentityactivation_2/Relu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџо
NoOpNoOp+^depthbatch/FusedBatchNormV3/ReadVariableOp-^depthbatch/FusedBatchNormV3/ReadVariableOp_1^depthbatch/ReadVariableOp^depthbatch/ReadVariableOp_1!^depthconv/BiasAdd/ReadVariableOp#^depthconv/depthwise/ReadVariableOp)^expbatch/FusedBatchNormV3/ReadVariableOp+^expbatch/FusedBatchNormV3/ReadVariableOp_1^expbatch/ReadVariableOp^expbatch/ReadVariableOp_1^expconv/BiasAdd/ReadVariableOp^expconv/Conv2D/ReadVariableOp*^projbatch/FusedBatchNormV3/ReadVariableOp,^projbatch/FusedBatchNormV3/ReadVariableOp_1^projbatch/ReadVariableOp^projbatch/ReadVariableOp_1 ^projconv/BiasAdd/ReadVariableOp^projconv/Conv2D/ReadVariableOp+^shortbatch/FusedBatchNormV3/ReadVariableOp-^shortbatch/FusedBatchNormV3/ReadVariableOp_1^shortbatch/ReadVariableOp^shortbatch/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 2X
*depthbatch/FusedBatchNormV3/ReadVariableOp*depthbatch/FusedBatchNormV3/ReadVariableOp2\
,depthbatch/FusedBatchNormV3/ReadVariableOp_1,depthbatch/FusedBatchNormV3/ReadVariableOp_126
depthbatch/ReadVariableOpdepthbatch/ReadVariableOp2:
depthbatch/ReadVariableOp_1depthbatch/ReadVariableOp_12D
 depthconv/BiasAdd/ReadVariableOp depthconv/BiasAdd/ReadVariableOp2H
"depthconv/depthwise/ReadVariableOp"depthconv/depthwise/ReadVariableOp2T
(expbatch/FusedBatchNormV3/ReadVariableOp(expbatch/FusedBatchNormV3/ReadVariableOp2X
*expbatch/FusedBatchNormV3/ReadVariableOp_1*expbatch/FusedBatchNormV3/ReadVariableOp_122
expbatch/ReadVariableOpexpbatch/ReadVariableOp26
expbatch/ReadVariableOp_1expbatch/ReadVariableOp_12@
expconv/BiasAdd/ReadVariableOpexpconv/BiasAdd/ReadVariableOp2>
expconv/Conv2D/ReadVariableOpexpconv/Conv2D/ReadVariableOp2V
)projbatch/FusedBatchNormV3/ReadVariableOp)projbatch/FusedBatchNormV3/ReadVariableOp2Z
+projbatch/FusedBatchNormV3/ReadVariableOp_1+projbatch/FusedBatchNormV3/ReadVariableOp_124
projbatch/ReadVariableOpprojbatch/ReadVariableOp28
projbatch/ReadVariableOp_1projbatch/ReadVariableOp_12B
projconv/BiasAdd/ReadVariableOpprojconv/BiasAdd/ReadVariableOp2@
projconv/Conv2D/ReadVariableOpprojconv/Conv2D/ReadVariableOp2X
*shortbatch/FusedBatchNormV3/ReadVariableOp*shortbatch/FusedBatchNormV3/ReadVariableOp2\
,shortbatch/FusedBatchNormV3/ReadVariableOp_1,shortbatch/FusedBatchNormV3/ReadVariableOp_126
shortbatch/ReadVariableOpshortbatch/ReadVariableOp2:
shortbatch/ReadVariableOp_1shortbatch/ReadVariableOp_1:R N
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameX
Р

E__inference_shortbatch_layer_call_and_return_conditional_losses_47741

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
љ
Х
*__inference_shortbatch_layer_call_fn_51172

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_shortbatch_layer_call_and_return_conditional_losses_47473
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ы
c
G__inference_activation_2_layer_call_and_return_conditional_losses_50299

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџppb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџpp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџpp:W S
/
_output_shapes
:џџџџџџџџџpp
 
_user_specified_nameinputs
љ
Г
D__inference_projbatch_layer_call_and_return_conditional_losses_47440

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
љ
Г
D__inference_projbatch_layer_call_and_return_conditional_losses_51159

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њ
Д
E__inference_shortbatch_layer_call_and_return_conditional_losses_51469

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њ
Д
E__inference_shortbatch_layer_call_and_return_conditional_losses_51221

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ј
В
C__inference_expbatch_layer_call_and_return_conditional_losses_47312

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ѓ
У
(__inference_expbatch_layer_call_fn_51247

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_expbatch_layer_call_and_return_conditional_losses_47580
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
њ
Д
E__inference_depthbatch_layer_call_and_return_conditional_losses_47376

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
цo
њ
F__inference_Bottleneck2_layer_call_and_return_conditional_losses_48437
x@
&expconv_conv2d_readvariableop_resource:05
'expconv_biasadd_readvariableop_resource:0.
 expbatch_readvariableop_resource:00
"expbatch_readvariableop_1_resource:0?
1expbatch_fusedbatchnormv3_readvariableop_resource:0A
3expbatch_fusedbatchnormv3_readvariableop_1_resource:0E
+depthconv_depthwise_readvariableop_resource:07
)depthconv_biasadd_readvariableop_resource:00
"depthbatch_readvariableop_resource:02
$depthbatch_readvariableop_1_resource:0A
3depthbatch_fusedbatchnormv3_readvariableop_resource:0C
5depthbatch_fusedbatchnormv3_readvariableop_1_resource:0A
'projconv_conv2d_readvariableop_resource:06
(projconv_biasadd_readvariableop_resource:/
!projbatch_readvariableop_resource:1
#projbatch_readvariableop_1_resource:@
2projbatch_fusedbatchnormv3_readvariableop_resource:B
4projbatch_fusedbatchnormv3_readvariableop_1_resource:0
"shortbatch_readvariableop_resource:2
$shortbatch_readvariableop_1_resource:A
3shortbatch_fusedbatchnormv3_readvariableop_resource:C
5shortbatch_fusedbatchnormv3_readvariableop_1_resource:
identityЂdepthbatch/AssignNewValueЂdepthbatch/AssignNewValue_1Ђ*depthbatch/FusedBatchNormV3/ReadVariableOpЂ,depthbatch/FusedBatchNormV3/ReadVariableOp_1Ђdepthbatch/ReadVariableOpЂdepthbatch/ReadVariableOp_1Ђ depthconv/BiasAdd/ReadVariableOpЂ"depthconv/depthwise/ReadVariableOpЂexpbatch/AssignNewValueЂexpbatch/AssignNewValue_1Ђ(expbatch/FusedBatchNormV3/ReadVariableOpЂ*expbatch/FusedBatchNormV3/ReadVariableOp_1Ђexpbatch/ReadVariableOpЂexpbatch/ReadVariableOp_1Ђexpconv/BiasAdd/ReadVariableOpЂexpconv/Conv2D/ReadVariableOpЂprojbatch/AssignNewValueЂprojbatch/AssignNewValue_1Ђ)projbatch/FusedBatchNormV3/ReadVariableOpЂ+projbatch/FusedBatchNormV3/ReadVariableOp_1Ђprojbatch/ReadVariableOpЂprojbatch/ReadVariableOp_1Ђprojconv/BiasAdd/ReadVariableOpЂprojconv/Conv2D/ReadVariableOpЂshortbatch/AssignNewValueЂshortbatch/AssignNewValue_1Ђ*shortbatch/FusedBatchNormV3/ReadVariableOpЂ,shortbatch/FusedBatchNormV3/ReadVariableOp_1Ђshortbatch/ReadVariableOpЂshortbatch/ReadVariableOp_1
expconv/Conv2D/ReadVariableOpReadVariableOp&expconv_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0Є
expconv/Conv2DConv2Dx%expconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides

expconv/BiasAdd/ReadVariableOpReadVariableOp'expconv_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
expconv/BiasAddBiasAddexpconv/Conv2D:output:0&expconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0t
expbatch/ReadVariableOpReadVariableOp expbatch_readvariableop_resource*
_output_shapes
:0*
dtype0x
expbatch/ReadVariableOp_1ReadVariableOp"expbatch_readvariableop_1_resource*
_output_shapes
:0*
dtype0
(expbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp1expbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
*expbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3expbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0
expbatch/FusedBatchNormV3FusedBatchNormV3expconv/BiasAdd:output:0expbatch/ReadVariableOp:value:0!expbatch/ReadVariableOp_1:value:00expbatch/FusedBatchNormV3/ReadVariableOp:value:02expbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<ъ
expbatch/AssignNewValueAssignVariableOp1expbatch_fusedbatchnormv3_readvariableop_resource&expbatch/FusedBatchNormV3:batch_mean:0)^expbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(є
expbatch/AssignNewValue_1AssignVariableOp3expbatch_fusedbatchnormv3_readvariableop_1_resource*expbatch/FusedBatchNormV3:batch_variance:0+^expbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(p
activation/ReluReluexpbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ0
"depthconv/depthwise/ReadVariableOpReadVariableOp+depthconv_depthwise_readvariableop_resource*&
_output_shapes
:0*
dtype0r
depthconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      r
!depthconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      й
depthconv/depthwiseDepthwiseConv2dNativeactivation/Relu:activations:0*depthconv/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides

 depthconv/BiasAdd/ReadVariableOpReadVariableOp)depthconv_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
depthconv/BiasAddBiasAdddepthconv/depthwise:output:0(depthconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0x
depthbatch/ReadVariableOpReadVariableOp"depthbatch_readvariableop_resource*
_output_shapes
:0*
dtype0|
depthbatch/ReadVariableOp_1ReadVariableOp$depthbatch_readvariableop_1_resource*
_output_shapes
:0*
dtype0
*depthbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp3depthbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
,depthbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5depthbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0
depthbatch/FusedBatchNormV3FusedBatchNormV3depthconv/BiasAdd:output:0!depthbatch/ReadVariableOp:value:0#depthbatch/ReadVariableOp_1:value:02depthbatch/FusedBatchNormV3/ReadVariableOp:value:04depthbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<ђ
depthbatch/AssignNewValueAssignVariableOp3depthbatch_fusedbatchnormv3_readvariableop_resource(depthbatch/FusedBatchNormV3:batch_mean:0+^depthbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ќ
depthbatch/AssignNewValue_1AssignVariableOp5depthbatch_fusedbatchnormv3_readvariableop_1_resource,depthbatch/FusedBatchNormV3:batch_variance:0-^depthbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(t
activation_1/ReluReludepthbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ0
projconv/Conv2D/ReadVariableOpReadVariableOp'projconv_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0Ф
projconv/Conv2DConv2Dactivation_1/Relu:activations:0&projconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

projconv/BiasAdd/ReadVariableOpReadVariableOp(projconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
projconv/BiasAddBiasAddprojconv/Conv2D:output:0'projconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџv
projbatch/ReadVariableOpReadVariableOp!projbatch_readvariableop_resource*
_output_shapes
:*
dtype0z
projbatch/ReadVariableOp_1ReadVariableOp#projbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
)projbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp2projbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
+projbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4projbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
projbatch/FusedBatchNormV3FusedBatchNormV3projconv/BiasAdd:output:0 projbatch/ReadVariableOp:value:0"projbatch/ReadVariableOp_1:value:01projbatch/FusedBatchNormV3/ReadVariableOp:value:03projbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<ю
projbatch/AssignNewValueAssignVariableOp2projbatch_fusedbatchnormv3_readvariableop_resource'projbatch/FusedBatchNormV3:batch_mean:0*^projbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ј
projbatch/AssignNewValue_1AssignVariableOp4projbatch_fusedbatchnormv3_readvariableop_1_resource+projbatch/FusedBatchNormV3:batch_variance:0,^projbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(x
shortbatch/ReadVariableOpReadVariableOp"shortbatch_readvariableop_resource*
_output_shapes
:*
dtype0|
shortbatch/ReadVariableOp_1ReadVariableOp$shortbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
*shortbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp3shortbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
,shortbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5shortbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0і
shortbatch/FusedBatchNormV3FusedBatchNormV3x!shortbatch/ReadVariableOp:value:0#shortbatch/ReadVariableOp_1:value:02shortbatch/FusedBatchNormV3/ReadVariableOp:value:04shortbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<ђ
shortbatch/AssignNewValueAssignVariableOp3shortbatch_fusedbatchnormv3_readvariableop_resource(shortbatch/FusedBatchNormV3:batch_mean:0+^shortbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ќ
shortbatch/AssignNewValue_1AssignVariableOp5shortbatch_fusedbatchnormv3_readvariableop_1_resource,shortbatch/FusedBatchNormV3:batch_variance:0-^shortbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
add/addAddV2projbatch/FusedBatchNormV3:y:0shortbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ`
activation_2/ReluReluadd/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџv
IdentityIdentityactivation_2/Relu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџР
NoOpNoOp^depthbatch/AssignNewValue^depthbatch/AssignNewValue_1+^depthbatch/FusedBatchNormV3/ReadVariableOp-^depthbatch/FusedBatchNormV3/ReadVariableOp_1^depthbatch/ReadVariableOp^depthbatch/ReadVariableOp_1!^depthconv/BiasAdd/ReadVariableOp#^depthconv/depthwise/ReadVariableOp^expbatch/AssignNewValue^expbatch/AssignNewValue_1)^expbatch/FusedBatchNormV3/ReadVariableOp+^expbatch/FusedBatchNormV3/ReadVariableOp_1^expbatch/ReadVariableOp^expbatch/ReadVariableOp_1^expconv/BiasAdd/ReadVariableOp^expconv/Conv2D/ReadVariableOp^projbatch/AssignNewValue^projbatch/AssignNewValue_1*^projbatch/FusedBatchNormV3/ReadVariableOp,^projbatch/FusedBatchNormV3/ReadVariableOp_1^projbatch/ReadVariableOp^projbatch/ReadVariableOp_1 ^projconv/BiasAdd/ReadVariableOp^projconv/Conv2D/ReadVariableOp^shortbatch/AssignNewValue^shortbatch/AssignNewValue_1+^shortbatch/FusedBatchNormV3/ReadVariableOp-^shortbatch/FusedBatchNormV3/ReadVariableOp_1^shortbatch/ReadVariableOp^shortbatch/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 26
depthbatch/AssignNewValuedepthbatch/AssignNewValue2:
depthbatch/AssignNewValue_1depthbatch/AssignNewValue_12X
*depthbatch/FusedBatchNormV3/ReadVariableOp*depthbatch/FusedBatchNormV3/ReadVariableOp2\
,depthbatch/FusedBatchNormV3/ReadVariableOp_1,depthbatch/FusedBatchNormV3/ReadVariableOp_126
depthbatch/ReadVariableOpdepthbatch/ReadVariableOp2:
depthbatch/ReadVariableOp_1depthbatch/ReadVariableOp_12D
 depthconv/BiasAdd/ReadVariableOp depthconv/BiasAdd/ReadVariableOp2H
"depthconv/depthwise/ReadVariableOp"depthconv/depthwise/ReadVariableOp22
expbatch/AssignNewValueexpbatch/AssignNewValue26
expbatch/AssignNewValue_1expbatch/AssignNewValue_12T
(expbatch/FusedBatchNormV3/ReadVariableOp(expbatch/FusedBatchNormV3/ReadVariableOp2X
*expbatch/FusedBatchNormV3/ReadVariableOp_1*expbatch/FusedBatchNormV3/ReadVariableOp_122
expbatch/ReadVariableOpexpbatch/ReadVariableOp26
expbatch/ReadVariableOp_1expbatch/ReadVariableOp_12@
expconv/BiasAdd/ReadVariableOpexpconv/BiasAdd/ReadVariableOp2>
expconv/Conv2D/ReadVariableOpexpconv/Conv2D/ReadVariableOp24
projbatch/AssignNewValueprojbatch/AssignNewValue28
projbatch/AssignNewValue_1projbatch/AssignNewValue_12V
)projbatch/FusedBatchNormV3/ReadVariableOp)projbatch/FusedBatchNormV3/ReadVariableOp2Z
+projbatch/FusedBatchNormV3/ReadVariableOp_1+projbatch/FusedBatchNormV3/ReadVariableOp_124
projbatch/ReadVariableOpprojbatch/ReadVariableOp28
projbatch/ReadVariableOp_1projbatch/ReadVariableOp_12B
projconv/BiasAdd/ReadVariableOpprojconv/BiasAdd/ReadVariableOp2@
projconv/Conv2D/ReadVariableOpprojconv/Conv2D/ReadVariableOp26
shortbatch/AssignNewValueshortbatch/AssignNewValue2:
shortbatch/AssignNewValue_1shortbatch/AssignNewValue_12X
*shortbatch/FusedBatchNormV3/ReadVariableOp*shortbatch/FusedBatchNormV3/ReadVariableOp2\
,shortbatch/FusedBatchNormV3/ReadVariableOp_1,shortbatch/FusedBatchNormV3/ReadVariableOp_126
shortbatch/ReadVariableOpshortbatch/ReadVariableOp2:
shortbatch/ReadVariableOp_1shortbatch/ReadVariableOp_1:R N
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameX
ї
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_50324

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ88c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ88"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ88:W S
/
_output_shapes
:џџџџџџџџџ88
 
_user_specified_nameinputs


є
B__inference_dense_2_layer_call_and_return_conditional_losses_50973

inputs1
matmul_readvariableop_resource:		-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:		*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
ј
В
C__inference_expbatch_layer_call_and_return_conditional_losses_51283

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
љ
Х
*__inference_shortbatch_layer_call_fn_51420

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_shortbatch_layer_call_and_return_conditional_losses_47741
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

b
)__inference_dropout_8_layer_call_fn_50925

inputs
identityЂStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_48298w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ж
K
/__inference_max_pooling2d_6_layer_call_fn_50304

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_47256
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ц
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_50953

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЧЅ
щ6
 __inference__wrapped_model_47183
input_3J
0model_2_conv2d_18_conv2d_readvariableop_resource:?
1model_2_conv2d_18_biasadd_readvariableop_resource:D
6model_2_batch_normalization_34_readvariableop_resource:F
8model_2_batch_normalization_34_readvariableop_1_resource:U
Gmodel_2_batch_normalization_34_fusedbatchnormv3_readvariableop_resource:W
Imodel_2_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:T
:model_2_bottleneck1_expconv_conv2d_readvariableop_resource:I
;model_2_bottleneck1_expconv_biasadd_readvariableop_resource:B
4model_2_bottleneck1_expbatch_readvariableop_resource:D
6model_2_bottleneck1_expbatch_readvariableop_1_resource:S
Emodel_2_bottleneck1_expbatch_fusedbatchnormv3_readvariableop_resource:U
Gmodel_2_bottleneck1_expbatch_fusedbatchnormv3_readvariableop_1_resource:Y
?model_2_bottleneck1_depthconv_depthwise_readvariableop_resource:K
=model_2_bottleneck1_depthconv_biasadd_readvariableop_resource:D
6model_2_bottleneck1_depthbatch_readvariableop_resource:F
8model_2_bottleneck1_depthbatch_readvariableop_1_resource:U
Gmodel_2_bottleneck1_depthbatch_fusedbatchnormv3_readvariableop_resource:W
Imodel_2_bottleneck1_depthbatch_fusedbatchnormv3_readvariableop_1_resource:U
;model_2_bottleneck1_projconv_conv2d_readvariableop_resource:J
<model_2_bottleneck1_projconv_biasadd_readvariableop_resource:C
5model_2_bottleneck1_projbatch_readvariableop_resource:E
7model_2_bottleneck1_projbatch_readvariableop_1_resource:T
Fmodel_2_bottleneck1_projbatch_fusedbatchnormv3_readvariableop_resource:V
Hmodel_2_bottleneck1_projbatch_fusedbatchnormv3_readvariableop_1_resource:D
6model_2_bottleneck1_shortbatch_readvariableop_resource:F
8model_2_bottleneck1_shortbatch_readvariableop_1_resource:U
Gmodel_2_bottleneck1_shortbatch_fusedbatchnormv3_readvariableop_resource:W
Imodel_2_bottleneck1_shortbatch_fusedbatchnormv3_readvariableop_1_resource:T
:model_2_bottleneck2_expconv_conv2d_readvariableop_resource:0I
;model_2_bottleneck2_expconv_biasadd_readvariableop_resource:0B
4model_2_bottleneck2_expbatch_readvariableop_resource:0D
6model_2_bottleneck2_expbatch_readvariableop_1_resource:0S
Emodel_2_bottleneck2_expbatch_fusedbatchnormv3_readvariableop_resource:0U
Gmodel_2_bottleneck2_expbatch_fusedbatchnormv3_readvariableop_1_resource:0Y
?model_2_bottleneck2_depthconv_depthwise_readvariableop_resource:0K
=model_2_bottleneck2_depthconv_biasadd_readvariableop_resource:0D
6model_2_bottleneck2_depthbatch_readvariableop_resource:0F
8model_2_bottleneck2_depthbatch_readvariableop_1_resource:0U
Gmodel_2_bottleneck2_depthbatch_fusedbatchnormv3_readvariableop_resource:0W
Imodel_2_bottleneck2_depthbatch_fusedbatchnormv3_readvariableop_1_resource:0U
;model_2_bottleneck2_projconv_conv2d_readvariableop_resource:0J
<model_2_bottleneck2_projconv_biasadd_readvariableop_resource:C
5model_2_bottleneck2_projbatch_readvariableop_resource:E
7model_2_bottleneck2_projbatch_readvariableop_1_resource:T
Fmodel_2_bottleneck2_projbatch_fusedbatchnormv3_readvariableop_resource:V
Hmodel_2_bottleneck2_projbatch_fusedbatchnormv3_readvariableop_1_resource:D
6model_2_bottleneck2_shortbatch_readvariableop_resource:F
8model_2_bottleneck2_shortbatch_readvariableop_1_resource:U
Gmodel_2_bottleneck2_shortbatch_fusedbatchnormv3_readvariableop_resource:W
Imodel_2_bottleneck2_shortbatch_fusedbatchnormv3_readvariableop_1_resource:A
.model_2_dense_2_matmul_readvariableop_resource:		=
/model_2_dense_2_biasadd_readvariableop_resource:
identityЂ>model_2/Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOpЂ@model_2/Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_1Ђ-model_2/Bottleneck1/depthbatch/ReadVariableOpЂ/model_2/Bottleneck1/depthbatch/ReadVariableOp_1Ђ4model_2/Bottleneck1/depthconv/BiasAdd/ReadVariableOpЂ6model_2/Bottleneck1/depthconv/depthwise/ReadVariableOpЂ<model_2/Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOpЂ>model_2/Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_1Ђ+model_2/Bottleneck1/expbatch/ReadVariableOpЂ-model_2/Bottleneck1/expbatch/ReadVariableOp_1Ђ2model_2/Bottleneck1/expconv/BiasAdd/ReadVariableOpЂ1model_2/Bottleneck1/expconv/Conv2D/ReadVariableOpЂ=model_2/Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOpЂ?model_2/Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_1Ђ,model_2/Bottleneck1/projbatch/ReadVariableOpЂ.model_2/Bottleneck1/projbatch/ReadVariableOp_1Ђ3model_2/Bottleneck1/projconv/BiasAdd/ReadVariableOpЂ2model_2/Bottleneck1/projconv/Conv2D/ReadVariableOpЂ>model_2/Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOpЂ@model_2/Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_1Ђ-model_2/Bottleneck1/shortbatch/ReadVariableOpЂ/model_2/Bottleneck1/shortbatch/ReadVariableOp_1Ђ>model_2/Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOpЂ@model_2/Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_1Ђ-model_2/Bottleneck2/depthbatch/ReadVariableOpЂ/model_2/Bottleneck2/depthbatch/ReadVariableOp_1Ђ4model_2/Bottleneck2/depthconv/BiasAdd/ReadVariableOpЂ6model_2/Bottleneck2/depthconv/depthwise/ReadVariableOpЂ<model_2/Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOpЂ>model_2/Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_1Ђ+model_2/Bottleneck2/expbatch/ReadVariableOpЂ-model_2/Bottleneck2/expbatch/ReadVariableOp_1Ђ2model_2/Bottleneck2/expconv/BiasAdd/ReadVariableOpЂ1model_2/Bottleneck2/expconv/Conv2D/ReadVariableOpЂ=model_2/Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOpЂ?model_2/Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_1Ђ,model_2/Bottleneck2/projbatch/ReadVariableOpЂ.model_2/Bottleneck2/projbatch/ReadVariableOp_1Ђ3model_2/Bottleneck2/projconv/BiasAdd/ReadVariableOpЂ2model_2/Bottleneck2/projconv/Conv2D/ReadVariableOpЂ>model_2/Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOpЂ@model_2/Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_1Ђ-model_2/Bottleneck2/shortbatch/ReadVariableOpЂ/model_2/Bottleneck2/shortbatch/ReadVariableOp_1Ђ>model_2/batch_normalization_34/FusedBatchNormV3/ReadVariableOpЂ@model_2/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1Ђ-model_2/batch_normalization_34/ReadVariableOpЂ/model_2/batch_normalization_34/ReadVariableOp_1Ђ(model_2/conv2d_18/BiasAdd/ReadVariableOpЂ'model_2/conv2d_18/Conv2D/ReadVariableOpЂ&model_2/dense_2/BiasAdd/ReadVariableOpЂ%model_2/dense_2/MatMul/ReadVariableOpo
model_2/tf.identity_2/IdentityIdentityinput_3*
T0*1
_output_shapes
:џџџџџџџџџрр_
model_2/rescaling_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;a
model_2/rescaling_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ј
model_2/rescaling_2/mulMul'model_2/tf.identity_2/Identity:output:0#model_2/rescaling_2/Cast/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџрр 
model_2/rescaling_2/addAddV2model_2/rescaling_2/mul:z:0%model_2/rescaling_2/Cast_1/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџрр 
'model_2/conv2d_18/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0в
model_2/conv2d_18/Conv2DConv2Dmodel_2/rescaling_2/add:z:0/model_2/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџpp*
paddingSAME*
strides

(model_2/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Г
model_2/conv2d_18/BiasAddBiasAdd!model_2/conv2d_18/Conv2D:output:00model_2/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџpp 
-model_2/batch_normalization_34/ReadVariableOpReadVariableOp6model_2_batch_normalization_34_readvariableop_resource*
_output_shapes
:*
dtype0Є
/model_2/batch_normalization_34/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_34_readvariableop_1_resource*
_output_shapes
:*
dtype0Т
>model_2/batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ц
@model_2/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0э
/model_2/batch_normalization_34/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_18/BiasAdd:output:05model_2/batch_normalization_34/ReadVariableOp:value:07model_2/batch_normalization_34/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџpp:::::*
epsilon%o:*
is_training( 
model_2/activation_2/ReluRelu3model_2/batch_normalization_34/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџppР
model_2/max_pooling2d_6/MaxPoolMaxPool'model_2/activation_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ88*
ksize
*
paddingVALID*
strides

model_2/dropout_6/IdentityIdentity(model_2/max_pooling2d_6/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ88Д
1model_2/Bottleneck1/expconv/Conv2D/ReadVariableOpReadVariableOp:model_2_bottleneck1_expconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ю
"model_2/Bottleneck1/expconv/Conv2DConv2D#model_2/dropout_6/Identity:output:09model_2/Bottleneck1/expconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides
Њ
2model_2/Bottleneck1/expconv/BiasAdd/ReadVariableOpReadVariableOp;model_2_bottleneck1_expconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0б
#model_2/Bottleneck1/expconv/BiasAddBiasAdd+model_2/Bottleneck1/expconv/Conv2D:output:0:model_2/Bottleneck1/expconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88
+model_2/Bottleneck1/expbatch/ReadVariableOpReadVariableOp4model_2_bottleneck1_expbatch_readvariableop_resource*
_output_shapes
:*
dtype0 
-model_2/Bottleneck1/expbatch/ReadVariableOp_1ReadVariableOp6model_2_bottleneck1_expbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0О
<model_2/Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOpReadVariableOpEmodel_2_bottleneck1_expbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Т
>model_2/Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGmodel_2_bottleneck1_expbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0э
-model_2/Bottleneck1/expbatch/FusedBatchNormV3FusedBatchNormV3,model_2/Bottleneck1/expconv/BiasAdd:output:03model_2/Bottleneck1/expbatch/ReadVariableOp:value:05model_2/Bottleneck1/expbatch/ReadVariableOp_1:value:0Dmodel_2/Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp:value:0Fmodel_2/Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
is_training( 
#model_2/Bottleneck1/activation/ReluRelu1model_2/Bottleneck1/expbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88О
6model_2/Bottleneck1/depthconv/depthwise/ReadVariableOpReadVariableOp?model_2_bottleneck1_depthconv_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0
-model_2/Bottleneck1/depthconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
5model_2/Bottleneck1/depthconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
'model_2/Bottleneck1/depthconv/depthwiseDepthwiseConv2dNative1model_2/Bottleneck1/activation/Relu:activations:0>model_2/Bottleneck1/depthconv/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides
Ў
4model_2/Bottleneck1/depthconv/BiasAdd/ReadVariableOpReadVariableOp=model_2_bottleneck1_depthconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0к
%model_2/Bottleneck1/depthconv/BiasAddBiasAdd0model_2/Bottleneck1/depthconv/depthwise:output:0<model_2/Bottleneck1/depthconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88 
-model_2/Bottleneck1/depthbatch/ReadVariableOpReadVariableOp6model_2_bottleneck1_depthbatch_readvariableop_resource*
_output_shapes
:*
dtype0Є
/model_2/Bottleneck1/depthbatch/ReadVariableOp_1ReadVariableOp8model_2_bottleneck1_depthbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0Т
>model_2/Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_bottleneck1_depthbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ц
@model_2/Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_bottleneck1_depthbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0љ
/model_2/Bottleneck1/depthbatch/FusedBatchNormV3FusedBatchNormV3.model_2/Bottleneck1/depthconv/BiasAdd:output:05model_2/Bottleneck1/depthbatch/ReadVariableOp:value:07model_2/Bottleneck1/depthbatch/ReadVariableOp_1:value:0Fmodel_2/Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
is_training( 
%model_2/Bottleneck1/activation_1/ReluRelu3model_2/Bottleneck1/depthbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88Ж
2model_2/Bottleneck1/projconv/Conv2D/ReadVariableOpReadVariableOp;model_2_bottleneck1_projconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#model_2/Bottleneck1/projconv/Conv2DConv2D3model_2/Bottleneck1/activation_1/Relu:activations:0:model_2/Bottleneck1/projconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides
Ќ
3model_2/Bottleneck1/projconv/BiasAdd/ReadVariableOpReadVariableOp<model_2_bottleneck1_projconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0д
$model_2/Bottleneck1/projconv/BiasAddBiasAdd,model_2/Bottleneck1/projconv/Conv2D:output:0;model_2/Bottleneck1/projconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88
,model_2/Bottleneck1/projbatch/ReadVariableOpReadVariableOp5model_2_bottleneck1_projbatch_readvariableop_resource*
_output_shapes
:*
dtype0Ђ
.model_2/Bottleneck1/projbatch/ReadVariableOp_1ReadVariableOp7model_2_bottleneck1_projbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0Р
=model_2/Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_bottleneck1_projbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ф
?model_2/Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_bottleneck1_projbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ѓ
.model_2/Bottleneck1/projbatch/FusedBatchNormV3FusedBatchNormV3-model_2/Bottleneck1/projconv/BiasAdd:output:04model_2/Bottleneck1/projbatch/ReadVariableOp:value:06model_2/Bottleneck1/projbatch/ReadVariableOp_1:value:0Emodel_2/Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
is_training(  
-model_2/Bottleneck1/shortbatch/ReadVariableOpReadVariableOp6model_2_bottleneck1_shortbatch_readvariableop_resource*
_output_shapes
:*
dtype0Є
/model_2/Bottleneck1/shortbatch/ReadVariableOp_1ReadVariableOp8model_2_bottleneck1_shortbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0Т
>model_2/Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_bottleneck1_shortbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ц
@model_2/Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_bottleneck1_shortbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ю
/model_2/Bottleneck1/shortbatch/FusedBatchNormV3FusedBatchNormV3#model_2/dropout_6/Identity:output:05model_2/Bottleneck1/shortbatch/ReadVariableOp:value:07model_2/Bottleneck1/shortbatch/ReadVariableOp_1:value:0Fmodel_2/Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
is_training( Ч
model_2/Bottleneck1/add/addAddV22model_2/Bottleneck1/projbatch/FusedBatchNormV3:y:03model_2/Bottleneck1/shortbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88
%model_2/Bottleneck1/activation_2/ReluRelumodel_2/Bottleneck1/add/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ88Ь
model_2/max_pooling2d_7/MaxPoolMaxPool3model_2/Bottleneck1/activation_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

model_2/dropout_7/IdentityIdentity(model_2/max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџД
1model_2/Bottleneck2/expconv/Conv2D/ReadVariableOpReadVariableOp:model_2_bottleneck2_expconv_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0ю
"model_2/Bottleneck2/expconv/Conv2DConv2D#model_2/dropout_7/Identity:output:09model_2/Bottleneck2/expconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides
Њ
2model_2/Bottleneck2/expconv/BiasAdd/ReadVariableOpReadVariableOp;model_2_bottleneck2_expconv_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0б
#model_2/Bottleneck2/expconv/BiasAddBiasAdd+model_2/Bottleneck2/expconv/Conv2D:output:0:model_2/Bottleneck2/expconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0
+model_2/Bottleneck2/expbatch/ReadVariableOpReadVariableOp4model_2_bottleneck2_expbatch_readvariableop_resource*
_output_shapes
:0*
dtype0 
-model_2/Bottleneck2/expbatch/ReadVariableOp_1ReadVariableOp6model_2_bottleneck2_expbatch_readvariableop_1_resource*
_output_shapes
:0*
dtype0О
<model_2/Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOpReadVariableOpEmodel_2_bottleneck2_expbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0Т
>model_2/Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGmodel_2_bottleneck2_expbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0э
-model_2/Bottleneck2/expbatch/FusedBatchNormV3FusedBatchNormV3,model_2/Bottleneck2/expconv/BiasAdd:output:03model_2/Bottleneck2/expbatch/ReadVariableOp:value:05model_2/Bottleneck2/expbatch/ReadVariableOp_1:value:0Dmodel_2/Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp:value:0Fmodel_2/Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( 
%model_2/Bottleneck2/activation_3/ReluRelu1model_2/Bottleneck2/expbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ0О
6model_2/Bottleneck2/depthconv/depthwise/ReadVariableOpReadVariableOp?model_2_bottleneck2_depthconv_depthwise_readvariableop_resource*&
_output_shapes
:0*
dtype0
-model_2/Bottleneck2/depthconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      
5model_2/Bottleneck2/depthconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
'model_2/Bottleneck2/depthconv/depthwiseDepthwiseConv2dNative3model_2/Bottleneck2/activation_3/Relu:activations:0>model_2/Bottleneck2/depthconv/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides
Ў
4model_2/Bottleneck2/depthconv/BiasAdd/ReadVariableOpReadVariableOp=model_2_bottleneck2_depthconv_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0к
%model_2/Bottleneck2/depthconv/BiasAddBiasAdd0model_2/Bottleneck2/depthconv/depthwise:output:0<model_2/Bottleneck2/depthconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0 
-model_2/Bottleneck2/depthbatch/ReadVariableOpReadVariableOp6model_2_bottleneck2_depthbatch_readvariableop_resource*
_output_shapes
:0*
dtype0Є
/model_2/Bottleneck2/depthbatch/ReadVariableOp_1ReadVariableOp8model_2_bottleneck2_depthbatch_readvariableop_1_resource*
_output_shapes
:0*
dtype0Т
>model_2/Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_bottleneck2_depthbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0Ц
@model_2/Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_bottleneck2_depthbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0љ
/model_2/Bottleneck2/depthbatch/FusedBatchNormV3FusedBatchNormV3.model_2/Bottleneck2/depthconv/BiasAdd:output:05model_2/Bottleneck2/depthbatch/ReadVariableOp:value:07model_2/Bottleneck2/depthbatch/ReadVariableOp_1:value:0Fmodel_2/Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( 
%model_2/Bottleneck2/activation_4/ReluRelu3model_2/Bottleneck2/depthbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ0Ж
2model_2/Bottleneck2/projconv/Conv2D/ReadVariableOpReadVariableOp;model_2_bottleneck2_projconv_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0
#model_2/Bottleneck2/projconv/Conv2DConv2D3model_2/Bottleneck2/activation_4/Relu:activations:0:model_2/Bottleneck2/projconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
Ќ
3model_2/Bottleneck2/projconv/BiasAdd/ReadVariableOpReadVariableOp<model_2_bottleneck2_projconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0д
$model_2/Bottleneck2/projconv/BiasAddBiasAdd,model_2/Bottleneck2/projconv/Conv2D:output:0;model_2/Bottleneck2/projconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
,model_2/Bottleneck2/projbatch/ReadVariableOpReadVariableOp5model_2_bottleneck2_projbatch_readvariableop_resource*
_output_shapes
:*
dtype0Ђ
.model_2/Bottleneck2/projbatch/ReadVariableOp_1ReadVariableOp7model_2_bottleneck2_projbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0Р
=model_2/Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_bottleneck2_projbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ф
?model_2/Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_bottleneck2_projbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ѓ
.model_2/Bottleneck2/projbatch/FusedBatchNormV3FusedBatchNormV3-model_2/Bottleneck2/projconv/BiasAdd:output:04model_2/Bottleneck2/projbatch/ReadVariableOp:value:06model_2/Bottleneck2/projbatch/ReadVariableOp_1:value:0Emodel_2/Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
is_training(  
-model_2/Bottleneck2/shortbatch/ReadVariableOpReadVariableOp6model_2_bottleneck2_shortbatch_readvariableop_resource*
_output_shapes
:*
dtype0Є
/model_2/Bottleneck2/shortbatch/ReadVariableOp_1ReadVariableOp8model_2_bottleneck2_shortbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0Т
>model_2/Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_bottleneck2_shortbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ц
@model_2/Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_bottleneck2_shortbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ю
/model_2/Bottleneck2/shortbatch/FusedBatchNormV3FusedBatchNormV3#model_2/dropout_7/Identity:output:05model_2/Bottleneck2/shortbatch/ReadVariableOp:value:07model_2/Bottleneck2/shortbatch/ReadVariableOp_1:value:0Fmodel_2/Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
is_training( Щ
model_2/Bottleneck2/add_1/addAddV22model_2/Bottleneck2/projbatch/FusedBatchNormV3:y:03model_2/Bottleneck2/shortbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ
%model_2/Bottleneck2/activation_5/ReluRelu!model_2/Bottleneck2/add_1/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџЬ
model_2/max_pooling2d_8/MaxPoolMaxPool3model_2/Bottleneck2/activation_5/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

model_2/dropout_8/IdentityIdentity(model_2/max_pooling2d_8/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџh
model_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  
model_2/flatten_2/ReshapeReshape#model_2/dropout_8/Identity:output:0 model_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	
%model_2/dense_2/MatMul/ReadVariableOpReadVariableOp.model_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:		*
dtype0Ѕ
model_2/dense_2/MatMulMatMul"model_2/flatten_2/Reshape:output:0-model_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&model_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0І
model_2/dense_2/BiasAddBiasAdd model_2/dense_2/MatMul:product:0.model_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџv
model_2/dense_2/SigmoidSigmoid model_2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
IdentityIdentitymodel_2/dense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџђ
NoOpNoOp?^model_2/Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOpA^model_2/Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_1.^model_2/Bottleneck1/depthbatch/ReadVariableOp0^model_2/Bottleneck1/depthbatch/ReadVariableOp_15^model_2/Bottleneck1/depthconv/BiasAdd/ReadVariableOp7^model_2/Bottleneck1/depthconv/depthwise/ReadVariableOp=^model_2/Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp?^model_2/Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_1,^model_2/Bottleneck1/expbatch/ReadVariableOp.^model_2/Bottleneck1/expbatch/ReadVariableOp_13^model_2/Bottleneck1/expconv/BiasAdd/ReadVariableOp2^model_2/Bottleneck1/expconv/Conv2D/ReadVariableOp>^model_2/Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp@^model_2/Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_1-^model_2/Bottleneck1/projbatch/ReadVariableOp/^model_2/Bottleneck1/projbatch/ReadVariableOp_14^model_2/Bottleneck1/projconv/BiasAdd/ReadVariableOp3^model_2/Bottleneck1/projconv/Conv2D/ReadVariableOp?^model_2/Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOpA^model_2/Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_1.^model_2/Bottleneck1/shortbatch/ReadVariableOp0^model_2/Bottleneck1/shortbatch/ReadVariableOp_1?^model_2/Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOpA^model_2/Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_1.^model_2/Bottleneck2/depthbatch/ReadVariableOp0^model_2/Bottleneck2/depthbatch/ReadVariableOp_15^model_2/Bottleneck2/depthconv/BiasAdd/ReadVariableOp7^model_2/Bottleneck2/depthconv/depthwise/ReadVariableOp=^model_2/Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp?^model_2/Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_1,^model_2/Bottleneck2/expbatch/ReadVariableOp.^model_2/Bottleneck2/expbatch/ReadVariableOp_13^model_2/Bottleneck2/expconv/BiasAdd/ReadVariableOp2^model_2/Bottleneck2/expconv/Conv2D/ReadVariableOp>^model_2/Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp@^model_2/Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_1-^model_2/Bottleneck2/projbatch/ReadVariableOp/^model_2/Bottleneck2/projbatch/ReadVariableOp_14^model_2/Bottleneck2/projconv/BiasAdd/ReadVariableOp3^model_2/Bottleneck2/projconv/Conv2D/ReadVariableOp?^model_2/Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOpA^model_2/Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_1.^model_2/Bottleneck2/shortbatch/ReadVariableOp0^model_2/Bottleneck2/shortbatch/ReadVariableOp_1?^model_2/batch_normalization_34/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_34/ReadVariableOp0^model_2/batch_normalization_34/ReadVariableOp_1)^model_2/conv2d_18/BiasAdd/ReadVariableOp(^model_2/conv2d_18/Conv2D/ReadVariableOp'^model_2/dense_2/BiasAdd/ReadVariableOp&^model_2/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџрр: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
>model_2/Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp>model_2/Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp2
@model_2/Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_1@model_2/Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_12^
-model_2/Bottleneck1/depthbatch/ReadVariableOp-model_2/Bottleneck1/depthbatch/ReadVariableOp2b
/model_2/Bottleneck1/depthbatch/ReadVariableOp_1/model_2/Bottleneck1/depthbatch/ReadVariableOp_12l
4model_2/Bottleneck1/depthconv/BiasAdd/ReadVariableOp4model_2/Bottleneck1/depthconv/BiasAdd/ReadVariableOp2p
6model_2/Bottleneck1/depthconv/depthwise/ReadVariableOp6model_2/Bottleneck1/depthconv/depthwise/ReadVariableOp2|
<model_2/Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp<model_2/Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp2
>model_2/Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_1>model_2/Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_12Z
+model_2/Bottleneck1/expbatch/ReadVariableOp+model_2/Bottleneck1/expbatch/ReadVariableOp2^
-model_2/Bottleneck1/expbatch/ReadVariableOp_1-model_2/Bottleneck1/expbatch/ReadVariableOp_12h
2model_2/Bottleneck1/expconv/BiasAdd/ReadVariableOp2model_2/Bottleneck1/expconv/BiasAdd/ReadVariableOp2f
1model_2/Bottleneck1/expconv/Conv2D/ReadVariableOp1model_2/Bottleneck1/expconv/Conv2D/ReadVariableOp2~
=model_2/Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp=model_2/Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp2
?model_2/Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_1?model_2/Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_12\
,model_2/Bottleneck1/projbatch/ReadVariableOp,model_2/Bottleneck1/projbatch/ReadVariableOp2`
.model_2/Bottleneck1/projbatch/ReadVariableOp_1.model_2/Bottleneck1/projbatch/ReadVariableOp_12j
3model_2/Bottleneck1/projconv/BiasAdd/ReadVariableOp3model_2/Bottleneck1/projconv/BiasAdd/ReadVariableOp2h
2model_2/Bottleneck1/projconv/Conv2D/ReadVariableOp2model_2/Bottleneck1/projconv/Conv2D/ReadVariableOp2
>model_2/Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp>model_2/Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp2
@model_2/Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_1@model_2/Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_12^
-model_2/Bottleneck1/shortbatch/ReadVariableOp-model_2/Bottleneck1/shortbatch/ReadVariableOp2b
/model_2/Bottleneck1/shortbatch/ReadVariableOp_1/model_2/Bottleneck1/shortbatch/ReadVariableOp_12
>model_2/Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp>model_2/Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp2
@model_2/Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_1@model_2/Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_12^
-model_2/Bottleneck2/depthbatch/ReadVariableOp-model_2/Bottleneck2/depthbatch/ReadVariableOp2b
/model_2/Bottleneck2/depthbatch/ReadVariableOp_1/model_2/Bottleneck2/depthbatch/ReadVariableOp_12l
4model_2/Bottleneck2/depthconv/BiasAdd/ReadVariableOp4model_2/Bottleneck2/depthconv/BiasAdd/ReadVariableOp2p
6model_2/Bottleneck2/depthconv/depthwise/ReadVariableOp6model_2/Bottleneck2/depthconv/depthwise/ReadVariableOp2|
<model_2/Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp<model_2/Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp2
>model_2/Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_1>model_2/Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_12Z
+model_2/Bottleneck2/expbatch/ReadVariableOp+model_2/Bottleneck2/expbatch/ReadVariableOp2^
-model_2/Bottleneck2/expbatch/ReadVariableOp_1-model_2/Bottleneck2/expbatch/ReadVariableOp_12h
2model_2/Bottleneck2/expconv/BiasAdd/ReadVariableOp2model_2/Bottleneck2/expconv/BiasAdd/ReadVariableOp2f
1model_2/Bottleneck2/expconv/Conv2D/ReadVariableOp1model_2/Bottleneck2/expconv/Conv2D/ReadVariableOp2~
=model_2/Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp=model_2/Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp2
?model_2/Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_1?model_2/Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_12\
,model_2/Bottleneck2/projbatch/ReadVariableOp,model_2/Bottleneck2/projbatch/ReadVariableOp2`
.model_2/Bottleneck2/projbatch/ReadVariableOp_1.model_2/Bottleneck2/projbatch/ReadVariableOp_12j
3model_2/Bottleneck2/projconv/BiasAdd/ReadVariableOp3model_2/Bottleneck2/projconv/BiasAdd/ReadVariableOp2h
2model_2/Bottleneck2/projconv/Conv2D/ReadVariableOp2model_2/Bottleneck2/projconv/Conv2D/ReadVariableOp2
>model_2/Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp>model_2/Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp2
@model_2/Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_1@model_2/Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_12^
-model_2/Bottleneck2/shortbatch/ReadVariableOp-model_2/Bottleneck2/shortbatch/ReadVariableOp2b
/model_2/Bottleneck2/shortbatch/ReadVariableOp_1/model_2/Bottleneck2/shortbatch/ReadVariableOp_12
>model_2/batch_normalization_34/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_34/FusedBatchNormV3/ReadVariableOp2
@model_2/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_34/ReadVariableOp-model_2/batch_normalization_34/ReadVariableOp2b
/model_2/batch_normalization_34/ReadVariableOp_1/model_2/batch_normalization_34/ReadVariableOp_12T
(model_2/conv2d_18/BiasAdd/ReadVariableOp(model_2/conv2d_18/BiasAdd/ReadVariableOp2R
'model_2/conv2d_18/Conv2D/ReadVariableOp'model_2/conv2d_18/Conv2D/ReadVariableOp2P
&model_2/dense_2/BiasAdd/ReadVariableOp&model_2/dense_2/BiasAdd/ReadVariableOp2N
%model_2/dense_2/MatMul/ReadVariableOp%model_2/dense_2/MatMul/ReadVariableOp:Z V
1
_output_shapes
:џџџџџџџџџрр
!
_user_specified_name	input_3
ї
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_47851

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ88c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ88"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ88:W S
/
_output_shapes
:џџџџџџџџџ88
 
_user_specified_nameinputs
Ћ

§
D__inference_conv2d_18_layer_call_and_return_conditional_losses_50227

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџpp*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџppg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџрр: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџрр
 
_user_specified_nameinputs
ѓ
У
(__inference_expbatch_layer_call_fn_50999

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_expbatch_layer_call_and_return_conditional_losses_47312
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ь
є
'__inference_model_2_layer_call_fn_49655

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:$

unknown_27:0

unknown_28:0

unknown_29:0

unknown_30:0

unknown_31:0

unknown_32:0$

unknown_33:0

unknown_34:0

unknown_35:0

unknown_36:0

unknown_37:0

unknown_38:0$

unknown_39:0

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:		

unknown_50:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_48155o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџрр: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџрр
 
_user_specified_nameinputs
ї
Х
*__inference_depthbatch_layer_call_fn_51309

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_depthbatch_layer_call_and_return_conditional_losses_47644
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
љ
Х
*__inference_depthbatch_layer_call_fn_51296

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_depthbatch_layer_call_and_return_conditional_losses_47613
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
Н
ѕ
'__inference_model_2_layer_call_fn_49185
input_3!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:$

unknown_27:0

unknown_28:0

unknown_29:0

unknown_30:0

unknown_31:0

unknown_32:0$

unknown_33:0

unknown_34:0

unknown_35:0

unknown_36:0

unknown_37:0

unknown_38:0$

unknown_39:0

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:		

unknown_50:
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*D
_read_only_resource_inputs&
$"	
 #$%&)*+,/034*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_48969o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџрр: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџрр
!
_user_specified_name	input_3
њ
Д
E__inference_depthbatch_layer_call_and_return_conditional_losses_47644

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
цo
њ
F__inference_Bottleneck2_layer_call_and_return_conditional_losses_50905
x@
&expconv_conv2d_readvariableop_resource:05
'expconv_biasadd_readvariableop_resource:0.
 expbatch_readvariableop_resource:00
"expbatch_readvariableop_1_resource:0?
1expbatch_fusedbatchnormv3_readvariableop_resource:0A
3expbatch_fusedbatchnormv3_readvariableop_1_resource:0E
+depthconv_depthwise_readvariableop_resource:07
)depthconv_biasadd_readvariableop_resource:00
"depthbatch_readvariableop_resource:02
$depthbatch_readvariableop_1_resource:0A
3depthbatch_fusedbatchnormv3_readvariableop_resource:0C
5depthbatch_fusedbatchnormv3_readvariableop_1_resource:0A
'projconv_conv2d_readvariableop_resource:06
(projconv_biasadd_readvariableop_resource:/
!projbatch_readvariableop_resource:1
#projbatch_readvariableop_1_resource:@
2projbatch_fusedbatchnormv3_readvariableop_resource:B
4projbatch_fusedbatchnormv3_readvariableop_1_resource:0
"shortbatch_readvariableop_resource:2
$shortbatch_readvariableop_1_resource:A
3shortbatch_fusedbatchnormv3_readvariableop_resource:C
5shortbatch_fusedbatchnormv3_readvariableop_1_resource:
identityЂdepthbatch/AssignNewValueЂdepthbatch/AssignNewValue_1Ђ*depthbatch/FusedBatchNormV3/ReadVariableOpЂ,depthbatch/FusedBatchNormV3/ReadVariableOp_1Ђdepthbatch/ReadVariableOpЂdepthbatch/ReadVariableOp_1Ђ depthconv/BiasAdd/ReadVariableOpЂ"depthconv/depthwise/ReadVariableOpЂexpbatch/AssignNewValueЂexpbatch/AssignNewValue_1Ђ(expbatch/FusedBatchNormV3/ReadVariableOpЂ*expbatch/FusedBatchNormV3/ReadVariableOp_1Ђexpbatch/ReadVariableOpЂexpbatch/ReadVariableOp_1Ђexpconv/BiasAdd/ReadVariableOpЂexpconv/Conv2D/ReadVariableOpЂprojbatch/AssignNewValueЂprojbatch/AssignNewValue_1Ђ)projbatch/FusedBatchNormV3/ReadVariableOpЂ+projbatch/FusedBatchNormV3/ReadVariableOp_1Ђprojbatch/ReadVariableOpЂprojbatch/ReadVariableOp_1Ђprojconv/BiasAdd/ReadVariableOpЂprojconv/Conv2D/ReadVariableOpЂshortbatch/AssignNewValueЂshortbatch/AssignNewValue_1Ђ*shortbatch/FusedBatchNormV3/ReadVariableOpЂ,shortbatch/FusedBatchNormV3/ReadVariableOp_1Ђshortbatch/ReadVariableOpЂshortbatch/ReadVariableOp_1
expconv/Conv2D/ReadVariableOpReadVariableOp&expconv_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0Є
expconv/Conv2DConv2Dx%expconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides

expconv/BiasAdd/ReadVariableOpReadVariableOp'expconv_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
expconv/BiasAddBiasAddexpconv/Conv2D:output:0&expconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0t
expbatch/ReadVariableOpReadVariableOp expbatch_readvariableop_resource*
_output_shapes
:0*
dtype0x
expbatch/ReadVariableOp_1ReadVariableOp"expbatch_readvariableop_1_resource*
_output_shapes
:0*
dtype0
(expbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp1expbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
*expbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3expbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0
expbatch/FusedBatchNormV3FusedBatchNormV3expconv/BiasAdd:output:0expbatch/ReadVariableOp:value:0!expbatch/ReadVariableOp_1:value:00expbatch/FusedBatchNormV3/ReadVariableOp:value:02expbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<ъ
expbatch/AssignNewValueAssignVariableOp1expbatch_fusedbatchnormv3_readvariableop_resource&expbatch/FusedBatchNormV3:batch_mean:0)^expbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(є
expbatch/AssignNewValue_1AssignVariableOp3expbatch_fusedbatchnormv3_readvariableop_1_resource*expbatch/FusedBatchNormV3:batch_variance:0+^expbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(p
activation/ReluReluexpbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ0
"depthconv/depthwise/ReadVariableOpReadVariableOp+depthconv_depthwise_readvariableop_resource*&
_output_shapes
:0*
dtype0r
depthconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      r
!depthconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      й
depthconv/depthwiseDepthwiseConv2dNativeactivation/Relu:activations:0*depthconv/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides

 depthconv/BiasAdd/ReadVariableOpReadVariableOp)depthconv_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
depthconv/BiasAddBiasAdddepthconv/depthwise:output:0(depthconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0x
depthbatch/ReadVariableOpReadVariableOp"depthbatch_readvariableop_resource*
_output_shapes
:0*
dtype0|
depthbatch/ReadVariableOp_1ReadVariableOp$depthbatch_readvariableop_1_resource*
_output_shapes
:0*
dtype0
*depthbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp3depthbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
,depthbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5depthbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0
depthbatch/FusedBatchNormV3FusedBatchNormV3depthconv/BiasAdd:output:0!depthbatch/ReadVariableOp:value:0#depthbatch/ReadVariableOp_1:value:02depthbatch/FusedBatchNormV3/ReadVariableOp:value:04depthbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<ђ
depthbatch/AssignNewValueAssignVariableOp3depthbatch_fusedbatchnormv3_readvariableop_resource(depthbatch/FusedBatchNormV3:batch_mean:0+^depthbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ќ
depthbatch/AssignNewValue_1AssignVariableOp5depthbatch_fusedbatchnormv3_readvariableop_1_resource,depthbatch/FusedBatchNormV3:batch_variance:0-^depthbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(t
activation_1/ReluReludepthbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ0
projconv/Conv2D/ReadVariableOpReadVariableOp'projconv_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0Ф
projconv/Conv2DConv2Dactivation_1/Relu:activations:0&projconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

projconv/BiasAdd/ReadVariableOpReadVariableOp(projconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
projconv/BiasAddBiasAddprojconv/Conv2D:output:0'projconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџv
projbatch/ReadVariableOpReadVariableOp!projbatch_readvariableop_resource*
_output_shapes
:*
dtype0z
projbatch/ReadVariableOp_1ReadVariableOp#projbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
)projbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp2projbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
+projbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4projbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
projbatch/FusedBatchNormV3FusedBatchNormV3projconv/BiasAdd:output:0 projbatch/ReadVariableOp:value:0"projbatch/ReadVariableOp_1:value:01projbatch/FusedBatchNormV3/ReadVariableOp:value:03projbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<ю
projbatch/AssignNewValueAssignVariableOp2projbatch_fusedbatchnormv3_readvariableop_resource'projbatch/FusedBatchNormV3:batch_mean:0*^projbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ј
projbatch/AssignNewValue_1AssignVariableOp4projbatch_fusedbatchnormv3_readvariableop_1_resource+projbatch/FusedBatchNormV3:batch_variance:0,^projbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(x
shortbatch/ReadVariableOpReadVariableOp"shortbatch_readvariableop_resource*
_output_shapes
:*
dtype0|
shortbatch/ReadVariableOp_1ReadVariableOp$shortbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
*shortbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp3shortbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
,shortbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5shortbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0і
shortbatch/FusedBatchNormV3FusedBatchNormV3x!shortbatch/ReadVariableOp:value:0#shortbatch/ReadVariableOp_1:value:02shortbatch/FusedBatchNormV3/ReadVariableOp:value:04shortbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<ђ
shortbatch/AssignNewValueAssignVariableOp3shortbatch_fusedbatchnormv3_readvariableop_resource(shortbatch/FusedBatchNormV3:batch_mean:0+^shortbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ќ
shortbatch/AssignNewValue_1AssignVariableOp5shortbatch_fusedbatchnormv3_readvariableop_1_resource,shortbatch/FusedBatchNormV3:batch_variance:0-^shortbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
add/addAddV2projbatch/FusedBatchNormV3:y:0shortbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ`
activation_2/ReluReluadd/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџv
IdentityIdentityactivation_2/Relu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџР
NoOpNoOp^depthbatch/AssignNewValue^depthbatch/AssignNewValue_1+^depthbatch/FusedBatchNormV3/ReadVariableOp-^depthbatch/FusedBatchNormV3/ReadVariableOp_1^depthbatch/ReadVariableOp^depthbatch/ReadVariableOp_1!^depthconv/BiasAdd/ReadVariableOp#^depthconv/depthwise/ReadVariableOp^expbatch/AssignNewValue^expbatch/AssignNewValue_1)^expbatch/FusedBatchNormV3/ReadVariableOp+^expbatch/FusedBatchNormV3/ReadVariableOp_1^expbatch/ReadVariableOp^expbatch/ReadVariableOp_1^expconv/BiasAdd/ReadVariableOp^expconv/Conv2D/ReadVariableOp^projbatch/AssignNewValue^projbatch/AssignNewValue_1*^projbatch/FusedBatchNormV3/ReadVariableOp,^projbatch/FusedBatchNormV3/ReadVariableOp_1^projbatch/ReadVariableOp^projbatch/ReadVariableOp_1 ^projconv/BiasAdd/ReadVariableOp^projconv/Conv2D/ReadVariableOp^shortbatch/AssignNewValue^shortbatch/AssignNewValue_1+^shortbatch/FusedBatchNormV3/ReadVariableOp-^shortbatch/FusedBatchNormV3/ReadVariableOp_1^shortbatch/ReadVariableOp^shortbatch/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 26
depthbatch/AssignNewValuedepthbatch/AssignNewValue2:
depthbatch/AssignNewValue_1depthbatch/AssignNewValue_12X
*depthbatch/FusedBatchNormV3/ReadVariableOp*depthbatch/FusedBatchNormV3/ReadVariableOp2\
,depthbatch/FusedBatchNormV3/ReadVariableOp_1,depthbatch/FusedBatchNormV3/ReadVariableOp_126
depthbatch/ReadVariableOpdepthbatch/ReadVariableOp2:
depthbatch/ReadVariableOp_1depthbatch/ReadVariableOp_12D
 depthconv/BiasAdd/ReadVariableOp depthconv/BiasAdd/ReadVariableOp2H
"depthconv/depthwise/ReadVariableOp"depthconv/depthwise/ReadVariableOp22
expbatch/AssignNewValueexpbatch/AssignNewValue26
expbatch/AssignNewValue_1expbatch/AssignNewValue_12T
(expbatch/FusedBatchNormV3/ReadVariableOp(expbatch/FusedBatchNormV3/ReadVariableOp2X
*expbatch/FusedBatchNormV3/ReadVariableOp_1*expbatch/FusedBatchNormV3/ReadVariableOp_122
expbatch/ReadVariableOpexpbatch/ReadVariableOp26
expbatch/ReadVariableOp_1expbatch/ReadVariableOp_12@
expconv/BiasAdd/ReadVariableOpexpconv/BiasAdd/ReadVariableOp2>
expconv/Conv2D/ReadVariableOpexpconv/Conv2D/ReadVariableOp24
projbatch/AssignNewValueprojbatch/AssignNewValue28
projbatch/AssignNewValue_1projbatch/AssignNewValue_12V
)projbatch/FusedBatchNormV3/ReadVariableOp)projbatch/FusedBatchNormV3/ReadVariableOp2Z
+projbatch/FusedBatchNormV3/ReadVariableOp_1+projbatch/FusedBatchNormV3/ReadVariableOp_124
projbatch/ReadVariableOpprojbatch/ReadVariableOp28
projbatch/ReadVariableOp_1projbatch/ReadVariableOp_12B
projconv/BiasAdd/ReadVariableOpprojconv/BiasAdd/ReadVariableOp2@
projconv/Conv2D/ReadVariableOpprojconv/Conv2D/ReadVariableOp26
shortbatch/AssignNewValueshortbatch/AssignNewValue2:
shortbatch/AssignNewValue_1shortbatch/AssignNewValue_12X
*shortbatch/FusedBatchNormV3/ReadVariableOp*shortbatch/FusedBatchNormV3/ReadVariableOp2\
,shortbatch/FusedBatchNormV3/ReadVariableOp_1,shortbatch/FusedBatchNormV3/ReadVariableOp_126
shortbatch/ReadVariableOpshortbatch/ReadVariableOp2:
shortbatch/ReadVariableOp_1shortbatch/ReadVariableOp_1:R N
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameX
ѕ
У
(__inference_expbatch_layer_call_fn_51234

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_expbatch_layer_call_and_return_conditional_losses_47549
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
юX

F__inference_Bottleneck1_layer_call_and_return_conditional_losses_50518
x@
&expconv_conv2d_readvariableop_resource:5
'expconv_biasadd_readvariableop_resource:.
 expbatch_readvariableop_resource:0
"expbatch_readvariableop_1_resource:?
1expbatch_fusedbatchnormv3_readvariableop_resource:A
3expbatch_fusedbatchnormv3_readvariableop_1_resource:E
+depthconv_depthwise_readvariableop_resource:7
)depthconv_biasadd_readvariableop_resource:0
"depthbatch_readvariableop_resource:2
$depthbatch_readvariableop_1_resource:A
3depthbatch_fusedbatchnormv3_readvariableop_resource:C
5depthbatch_fusedbatchnormv3_readvariableop_1_resource:A
'projconv_conv2d_readvariableop_resource:6
(projconv_biasadd_readvariableop_resource:/
!projbatch_readvariableop_resource:1
#projbatch_readvariableop_1_resource:@
2projbatch_fusedbatchnormv3_readvariableop_resource:B
4projbatch_fusedbatchnormv3_readvariableop_1_resource:0
"shortbatch_readvariableop_resource:2
$shortbatch_readvariableop_1_resource:A
3shortbatch_fusedbatchnormv3_readvariableop_resource:C
5shortbatch_fusedbatchnormv3_readvariableop_1_resource:
identityЂ*depthbatch/FusedBatchNormV3/ReadVariableOpЂ,depthbatch/FusedBatchNormV3/ReadVariableOp_1Ђdepthbatch/ReadVariableOpЂdepthbatch/ReadVariableOp_1Ђ depthconv/BiasAdd/ReadVariableOpЂ"depthconv/depthwise/ReadVariableOpЂ(expbatch/FusedBatchNormV3/ReadVariableOpЂ*expbatch/FusedBatchNormV3/ReadVariableOp_1Ђexpbatch/ReadVariableOpЂexpbatch/ReadVariableOp_1Ђexpconv/BiasAdd/ReadVariableOpЂexpconv/Conv2D/ReadVariableOpЂ)projbatch/FusedBatchNormV3/ReadVariableOpЂ+projbatch/FusedBatchNormV3/ReadVariableOp_1Ђprojbatch/ReadVariableOpЂprojbatch/ReadVariableOp_1Ђprojconv/BiasAdd/ReadVariableOpЂprojconv/Conv2D/ReadVariableOpЂ*shortbatch/FusedBatchNormV3/ReadVariableOpЂ,shortbatch/FusedBatchNormV3/ReadVariableOp_1Ђshortbatch/ReadVariableOpЂshortbatch/ReadVariableOp_1
expconv/Conv2D/ReadVariableOpReadVariableOp&expconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Є
expconv/Conv2DConv2Dx%expconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

expconv/BiasAdd/ReadVariableOpReadVariableOp'expconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
expconv/BiasAddBiasAddexpconv/Conv2D:output:0&expconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88t
expbatch/ReadVariableOpReadVariableOp expbatch_readvariableop_resource*
_output_shapes
:*
dtype0x
expbatch/ReadVariableOp_1ReadVariableOp"expbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
(expbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp1expbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
*expbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3expbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ѕ
expbatch/FusedBatchNormV3FusedBatchNormV3expconv/BiasAdd:output:0expbatch/ReadVariableOp:value:0!expbatch/ReadVariableOp_1:value:00expbatch/FusedBatchNormV3/ReadVariableOp:value:02expbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
is_training( p
activation/ReluReluexpbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88
"depthconv/depthwise/ReadVariableOpReadVariableOp+depthconv_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0r
depthconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            r
!depthconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      й
depthconv/depthwiseDepthwiseConv2dNativeactivation/Relu:activations:0*depthconv/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

 depthconv/BiasAdd/ReadVariableOpReadVariableOp)depthconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
depthconv/BiasAddBiasAdddepthconv/depthwise:output:0(depthconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88x
depthbatch/ReadVariableOpReadVariableOp"depthbatch_readvariableop_resource*
_output_shapes
:*
dtype0|
depthbatch/ReadVariableOp_1ReadVariableOp$depthbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
*depthbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp3depthbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
,depthbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5depthbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
depthbatch/FusedBatchNormV3FusedBatchNormV3depthconv/BiasAdd:output:0!depthbatch/ReadVariableOp:value:0#depthbatch/ReadVariableOp_1:value:02depthbatch/FusedBatchNormV3/ReadVariableOp:value:04depthbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
is_training( t
activation_1/ReluReludepthbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88
projconv/Conv2D/ReadVariableOpReadVariableOp'projconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
projconv/Conv2DConv2Dactivation_1/Relu:activations:0&projconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

projconv/BiasAdd/ReadVariableOpReadVariableOp(projconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
projconv/BiasAddBiasAddprojconv/Conv2D:output:0'projconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88v
projbatch/ReadVariableOpReadVariableOp!projbatch_readvariableop_resource*
_output_shapes
:*
dtype0z
projbatch/ReadVariableOp_1ReadVariableOp#projbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
)projbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp2projbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
+projbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4projbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ћ
projbatch/FusedBatchNormV3FusedBatchNormV3projconv/BiasAdd:output:0 projbatch/ReadVariableOp:value:0"projbatch/ReadVariableOp_1:value:01projbatch/FusedBatchNormV3/ReadVariableOp:value:03projbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
is_training( x
shortbatch/ReadVariableOpReadVariableOp"shortbatch_readvariableop_resource*
_output_shapes
:*
dtype0|
shortbatch/ReadVariableOp_1ReadVariableOp$shortbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
*shortbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp3shortbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
,shortbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5shortbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ш
shortbatch/FusedBatchNormV3FusedBatchNormV3x!shortbatch/ReadVariableOp:value:0#shortbatch/ReadVariableOp_1:value:02shortbatch/FusedBatchNormV3/ReadVariableOp:value:04shortbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
is_training( 
add/addAddV2projbatch/FusedBatchNormV3:y:0shortbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88`
activation_2/ReluReluadd/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ88v
IdentityIdentityactivation_2/Relu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ88о
NoOpNoOp+^depthbatch/FusedBatchNormV3/ReadVariableOp-^depthbatch/FusedBatchNormV3/ReadVariableOp_1^depthbatch/ReadVariableOp^depthbatch/ReadVariableOp_1!^depthconv/BiasAdd/ReadVariableOp#^depthconv/depthwise/ReadVariableOp)^expbatch/FusedBatchNormV3/ReadVariableOp+^expbatch/FusedBatchNormV3/ReadVariableOp_1^expbatch/ReadVariableOp^expbatch/ReadVariableOp_1^expconv/BiasAdd/ReadVariableOp^expconv/Conv2D/ReadVariableOp*^projbatch/FusedBatchNormV3/ReadVariableOp,^projbatch/FusedBatchNormV3/ReadVariableOp_1^projbatch/ReadVariableOp^projbatch/ReadVariableOp_1 ^projconv/BiasAdd/ReadVariableOp^projconv/Conv2D/ReadVariableOp+^shortbatch/FusedBatchNormV3/ReadVariableOp-^shortbatch/FusedBatchNormV3/ReadVariableOp_1^shortbatch/ReadVariableOp^shortbatch/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:џџџџџџџџџ88: : : : : : : : : : : : : : : : : : : : : : 2X
*depthbatch/FusedBatchNormV3/ReadVariableOp*depthbatch/FusedBatchNormV3/ReadVariableOp2\
,depthbatch/FusedBatchNormV3/ReadVariableOp_1,depthbatch/FusedBatchNormV3/ReadVariableOp_126
depthbatch/ReadVariableOpdepthbatch/ReadVariableOp2:
depthbatch/ReadVariableOp_1depthbatch/ReadVariableOp_12D
 depthconv/BiasAdd/ReadVariableOp depthconv/BiasAdd/ReadVariableOp2H
"depthconv/depthwise/ReadVariableOp"depthconv/depthwise/ReadVariableOp2T
(expbatch/FusedBatchNormV3/ReadVariableOp(expbatch/FusedBatchNormV3/ReadVariableOp2X
*expbatch/FusedBatchNormV3/ReadVariableOp_1*expbatch/FusedBatchNormV3/ReadVariableOp_122
expbatch/ReadVariableOpexpbatch/ReadVariableOp26
expbatch/ReadVariableOp_1expbatch/ReadVariableOp_12@
expconv/BiasAdd/ReadVariableOpexpconv/BiasAdd/ReadVariableOp2>
expconv/Conv2D/ReadVariableOpexpconv/Conv2D/ReadVariableOp2V
)projbatch/FusedBatchNormV3/ReadVariableOp)projbatch/FusedBatchNormV3/ReadVariableOp2Z
+projbatch/FusedBatchNormV3/ReadVariableOp_1+projbatch/FusedBatchNormV3/ReadVariableOp_124
projbatch/ReadVariableOpprojbatch/ReadVariableOp28
projbatch/ReadVariableOp_1projbatch/ReadVariableOp_12B
projconv/BiasAdd/ReadVariableOpprojconv/BiasAdd/ReadVariableOp2@
projconv/Conv2D/ReadVariableOpprojconv/Conv2D/ReadVariableOp2X
*shortbatch/FusedBatchNormV3/ReadVariableOp*shortbatch/FusedBatchNormV3/ReadVariableOp2\
,shortbatch/FusedBatchNormV3/ReadVariableOp_1,shortbatch/FusedBatchNormV3/ReadVariableOp_126
shortbatch/ReadVariableOpshortbatch/ReadVariableOp2:
shortbatch/ReadVariableOp_1shortbatch/ReadVariableOp_1:R N
/
_output_shapes
:џџџџџџџџџ88

_user_specified_nameX

Ъ0
B__inference_model_2_layer_call_and_return_conditional_losses_49969

inputsB
(conv2d_18_conv2d_readvariableop_resource:7
)conv2d_18_biasadd_readvariableop_resource:<
.batch_normalization_34_readvariableop_resource:>
0batch_normalization_34_readvariableop_1_resource:M
?batch_normalization_34_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:L
2bottleneck1_expconv_conv2d_readvariableop_resource:A
3bottleneck1_expconv_biasadd_readvariableop_resource::
,bottleneck1_expbatch_readvariableop_resource:<
.bottleneck1_expbatch_readvariableop_1_resource:K
=bottleneck1_expbatch_fusedbatchnormv3_readvariableop_resource:M
?bottleneck1_expbatch_fusedbatchnormv3_readvariableop_1_resource:Q
7bottleneck1_depthconv_depthwise_readvariableop_resource:C
5bottleneck1_depthconv_biasadd_readvariableop_resource:<
.bottleneck1_depthbatch_readvariableop_resource:>
0bottleneck1_depthbatch_readvariableop_1_resource:M
?bottleneck1_depthbatch_fusedbatchnormv3_readvariableop_resource:O
Abottleneck1_depthbatch_fusedbatchnormv3_readvariableop_1_resource:M
3bottleneck1_projconv_conv2d_readvariableop_resource:B
4bottleneck1_projconv_biasadd_readvariableop_resource:;
-bottleneck1_projbatch_readvariableop_resource:=
/bottleneck1_projbatch_readvariableop_1_resource:L
>bottleneck1_projbatch_fusedbatchnormv3_readvariableop_resource:N
@bottleneck1_projbatch_fusedbatchnormv3_readvariableop_1_resource:<
.bottleneck1_shortbatch_readvariableop_resource:>
0bottleneck1_shortbatch_readvariableop_1_resource:M
?bottleneck1_shortbatch_fusedbatchnormv3_readvariableop_resource:O
Abottleneck1_shortbatch_fusedbatchnormv3_readvariableop_1_resource:L
2bottleneck2_expconv_conv2d_readvariableop_resource:0A
3bottleneck2_expconv_biasadd_readvariableop_resource:0:
,bottleneck2_expbatch_readvariableop_resource:0<
.bottleneck2_expbatch_readvariableop_1_resource:0K
=bottleneck2_expbatch_fusedbatchnormv3_readvariableop_resource:0M
?bottleneck2_expbatch_fusedbatchnormv3_readvariableop_1_resource:0Q
7bottleneck2_depthconv_depthwise_readvariableop_resource:0C
5bottleneck2_depthconv_biasadd_readvariableop_resource:0<
.bottleneck2_depthbatch_readvariableop_resource:0>
0bottleneck2_depthbatch_readvariableop_1_resource:0M
?bottleneck2_depthbatch_fusedbatchnormv3_readvariableop_resource:0O
Abottleneck2_depthbatch_fusedbatchnormv3_readvariableop_1_resource:0M
3bottleneck2_projconv_conv2d_readvariableop_resource:0B
4bottleneck2_projconv_biasadd_readvariableop_resource:;
-bottleneck2_projbatch_readvariableop_resource:=
/bottleneck2_projbatch_readvariableop_1_resource:L
>bottleneck2_projbatch_fusedbatchnormv3_readvariableop_resource:N
@bottleneck2_projbatch_fusedbatchnormv3_readvariableop_1_resource:<
.bottleneck2_shortbatch_readvariableop_resource:>
0bottleneck2_shortbatch_readvariableop_1_resource:M
?bottleneck2_shortbatch_fusedbatchnormv3_readvariableop_resource:O
Abottleneck2_shortbatch_fusedbatchnormv3_readvariableop_1_resource:9
&dense_2_matmul_readvariableop_resource:		5
'dense_2_biasadd_readvariableop_resource:
identityЂ6Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOpЂ8Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_1Ђ%Bottleneck1/depthbatch/ReadVariableOpЂ'Bottleneck1/depthbatch/ReadVariableOp_1Ђ,Bottleneck1/depthconv/BiasAdd/ReadVariableOpЂ.Bottleneck1/depthconv/depthwise/ReadVariableOpЂ4Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOpЂ6Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_1Ђ#Bottleneck1/expbatch/ReadVariableOpЂ%Bottleneck1/expbatch/ReadVariableOp_1Ђ*Bottleneck1/expconv/BiasAdd/ReadVariableOpЂ)Bottleneck1/expconv/Conv2D/ReadVariableOpЂ5Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOpЂ7Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_1Ђ$Bottleneck1/projbatch/ReadVariableOpЂ&Bottleneck1/projbatch/ReadVariableOp_1Ђ+Bottleneck1/projconv/BiasAdd/ReadVariableOpЂ*Bottleneck1/projconv/Conv2D/ReadVariableOpЂ6Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOpЂ8Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_1Ђ%Bottleneck1/shortbatch/ReadVariableOpЂ'Bottleneck1/shortbatch/ReadVariableOp_1Ђ6Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOpЂ8Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_1Ђ%Bottleneck2/depthbatch/ReadVariableOpЂ'Bottleneck2/depthbatch/ReadVariableOp_1Ђ,Bottleneck2/depthconv/BiasAdd/ReadVariableOpЂ.Bottleneck2/depthconv/depthwise/ReadVariableOpЂ4Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOpЂ6Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_1Ђ#Bottleneck2/expbatch/ReadVariableOpЂ%Bottleneck2/expbatch/ReadVariableOp_1Ђ*Bottleneck2/expconv/BiasAdd/ReadVariableOpЂ)Bottleneck2/expconv/Conv2D/ReadVariableOpЂ5Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOpЂ7Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_1Ђ$Bottleneck2/projbatch/ReadVariableOpЂ&Bottleneck2/projbatch/ReadVariableOp_1Ђ+Bottleneck2/projconv/BiasAdd/ReadVariableOpЂ*Bottleneck2/projconv/Conv2D/ReadVariableOpЂ6Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOpЂ8Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_1Ђ%Bottleneck2/shortbatch/ReadVariableOpЂ'Bottleneck2/shortbatch/ReadVariableOp_1Ђ6batch_normalization_34/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_34/ReadVariableOpЂ'batch_normalization_34/ReadVariableOp_1Ђ conv2d_18/BiasAdd/ReadVariableOpЂconv2d_18/Conv2D/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpf
tf.identity_2/IdentityIdentityinputs*
T0*1
_output_shapes
:џџџџџџџџџррW
rescaling_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;Y
rescaling_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
rescaling_2/mulMultf.identity_2/Identity:output:0rescaling_2/Cast/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџрр
rescaling_2/addAddV2rescaling_2/mul:z:0rescaling_2/Cast_1/x:output:0*
T0*1
_output_shapes
:џџџџџџџџџрр
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0К
conv2d_18/Conv2DConv2Drescaling_2/add:z:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџpp*
paddingSAME*
strides

 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџpp
%batch_normalization_34/ReadVariableOpReadVariableOp.batch_normalization_34_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_34/ReadVariableOp_1ReadVariableOp0batch_normalization_34_readvariableop_1_resource*
_output_shapes
:*
dtype0В
6batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ж
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Н
'batch_normalization_34/FusedBatchNormV3FusedBatchNormV3conv2d_18/BiasAdd:output:0-batch_normalization_34/ReadVariableOp:value:0/batch_normalization_34/ReadVariableOp_1:value:0>batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџpp:::::*
epsilon%o:*
is_training( 
activation_2/ReluRelu+batch_normalization_34/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџppА
max_pooling2d_6/MaxPoolMaxPoolactivation_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ88*
ksize
*
paddingVALID*
strides
z
dropout_6/IdentityIdentity max_pooling2d_6/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ88Є
)Bottleneck1/expconv/Conv2D/ReadVariableOpReadVariableOp2bottleneck1_expconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ж
Bottleneck1/expconv/Conv2DConv2Ddropout_6/Identity:output:01Bottleneck1/expconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

*Bottleneck1/expconv/BiasAdd/ReadVariableOpReadVariableOp3bottleneck1_expconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Й
Bottleneck1/expconv/BiasAddBiasAdd#Bottleneck1/expconv/Conv2D:output:02Bottleneck1/expconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88
#Bottleneck1/expbatch/ReadVariableOpReadVariableOp,bottleneck1_expbatch_readvariableop_resource*
_output_shapes
:*
dtype0
%Bottleneck1/expbatch/ReadVariableOp_1ReadVariableOp.bottleneck1_expbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0Ў
4Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp=bottleneck1_expbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0В
6Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp?bottleneck1_expbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Н
%Bottleneck1/expbatch/FusedBatchNormV3FusedBatchNormV3$Bottleneck1/expconv/BiasAdd:output:0+Bottleneck1/expbatch/ReadVariableOp:value:0-Bottleneck1/expbatch/ReadVariableOp_1:value:0<Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp:value:0>Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
is_training( 
Bottleneck1/activation/ReluRelu)Bottleneck1/expbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88Ў
.Bottleneck1/depthconv/depthwise/ReadVariableOpReadVariableOp7bottleneck1_depthconv_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0~
%Bottleneck1/depthconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            ~
-Bottleneck1/depthconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      §
Bottleneck1/depthconv/depthwiseDepthwiseConv2dNative)Bottleneck1/activation/Relu:activations:06Bottleneck1/depthconv/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

,Bottleneck1/depthconv/BiasAdd/ReadVariableOpReadVariableOp5bottleneck1_depthconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Т
Bottleneck1/depthconv/BiasAddBiasAdd(Bottleneck1/depthconv/depthwise:output:04Bottleneck1/depthconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88
%Bottleneck1/depthbatch/ReadVariableOpReadVariableOp.bottleneck1_depthbatch_readvariableop_resource*
_output_shapes
:*
dtype0
'Bottleneck1/depthbatch/ReadVariableOp_1ReadVariableOp0bottleneck1_depthbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0В
6Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp?bottleneck1_depthbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ж
8Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbottleneck1_depthbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Щ
'Bottleneck1/depthbatch/FusedBatchNormV3FusedBatchNormV3&Bottleneck1/depthconv/BiasAdd:output:0-Bottleneck1/depthbatch/ReadVariableOp:value:0/Bottleneck1/depthbatch/ReadVariableOp_1:value:0>Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp:value:0@Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
is_training( 
Bottleneck1/activation_1/ReluRelu+Bottleneck1/depthbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88І
*Bottleneck1/projconv/Conv2D/ReadVariableOpReadVariableOp3bottleneck1_projconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ш
Bottleneck1/projconv/Conv2DConv2D+Bottleneck1/activation_1/Relu:activations:02Bottleneck1/projconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

+Bottleneck1/projconv/BiasAdd/ReadVariableOpReadVariableOp4bottleneck1_projconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0М
Bottleneck1/projconv/BiasAddBiasAdd$Bottleneck1/projconv/Conv2D:output:03Bottleneck1/projconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88
$Bottleneck1/projbatch/ReadVariableOpReadVariableOp-bottleneck1_projbatch_readvariableop_resource*
_output_shapes
:*
dtype0
&Bottleneck1/projbatch/ReadVariableOp_1ReadVariableOp/bottleneck1_projbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0А
5Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp>bottleneck1_projbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Д
7Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@bottleneck1_projbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0У
&Bottleneck1/projbatch/FusedBatchNormV3FusedBatchNormV3%Bottleneck1/projconv/BiasAdd:output:0,Bottleneck1/projbatch/ReadVariableOp:value:0.Bottleneck1/projbatch/ReadVariableOp_1:value:0=Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp:value:0?Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
is_training( 
%Bottleneck1/shortbatch/ReadVariableOpReadVariableOp.bottleneck1_shortbatch_readvariableop_resource*
_output_shapes
:*
dtype0
'Bottleneck1/shortbatch/ReadVariableOp_1ReadVariableOp0bottleneck1_shortbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0В
6Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp?bottleneck1_shortbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ж
8Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbottleneck1_shortbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0О
'Bottleneck1/shortbatch/FusedBatchNormV3FusedBatchNormV3dropout_6/Identity:output:0-Bottleneck1/shortbatch/ReadVariableOp:value:0/Bottleneck1/shortbatch/ReadVariableOp_1:value:0>Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp:value:0@Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
is_training( Џ
Bottleneck1/add/addAddV2*Bottleneck1/projbatch/FusedBatchNormV3:y:0+Bottleneck1/shortbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88x
Bottleneck1/activation_2/ReluReluBottleneck1/add/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ88М
max_pooling2d_7/MaxPoolMaxPool+Bottleneck1/activation_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
z
dropout_7/IdentityIdentity max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџЄ
)Bottleneck2/expconv/Conv2D/ReadVariableOpReadVariableOp2bottleneck2_expconv_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0ж
Bottleneck2/expconv/Conv2DConv2Ddropout_7/Identity:output:01Bottleneck2/expconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides

*Bottleneck2/expconv/BiasAdd/ReadVariableOpReadVariableOp3bottleneck2_expconv_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0Й
Bottleneck2/expconv/BiasAddBiasAdd#Bottleneck2/expconv/Conv2D:output:02Bottleneck2/expconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0
#Bottleneck2/expbatch/ReadVariableOpReadVariableOp,bottleneck2_expbatch_readvariableop_resource*
_output_shapes
:0*
dtype0
%Bottleneck2/expbatch/ReadVariableOp_1ReadVariableOp.bottleneck2_expbatch_readvariableop_1_resource*
_output_shapes
:0*
dtype0Ў
4Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp=bottleneck2_expbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0В
6Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp?bottleneck2_expbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0Н
%Bottleneck2/expbatch/FusedBatchNormV3FusedBatchNormV3$Bottleneck2/expconv/BiasAdd:output:0+Bottleneck2/expbatch/ReadVariableOp:value:0-Bottleneck2/expbatch/ReadVariableOp_1:value:0<Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp:value:0>Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( 
Bottleneck2/activation_3/ReluRelu)Bottleneck2/expbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ0Ў
.Bottleneck2/depthconv/depthwise/ReadVariableOpReadVariableOp7bottleneck2_depthconv_depthwise_readvariableop_resource*&
_output_shapes
:0*
dtype0~
%Bottleneck2/depthconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      0      ~
-Bottleneck2/depthconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      џ
Bottleneck2/depthconv/depthwiseDepthwiseConv2dNative+Bottleneck2/activation_3/Relu:activations:06Bottleneck2/depthconv/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides

,Bottleneck2/depthconv/BiasAdd/ReadVariableOpReadVariableOp5bottleneck2_depthconv_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0Т
Bottleneck2/depthconv/BiasAddBiasAdd(Bottleneck2/depthconv/depthwise:output:04Bottleneck2/depthconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ0
%Bottleneck2/depthbatch/ReadVariableOpReadVariableOp.bottleneck2_depthbatch_readvariableop_resource*
_output_shapes
:0*
dtype0
'Bottleneck2/depthbatch/ReadVariableOp_1ReadVariableOp0bottleneck2_depthbatch_readvariableop_1_resource*
_output_shapes
:0*
dtype0В
6Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp?bottleneck2_depthbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0Ж
8Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbottleneck2_depthbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0Щ
'Bottleneck2/depthbatch/FusedBatchNormV3FusedBatchNormV3&Bottleneck2/depthconv/BiasAdd:output:0-Bottleneck2/depthbatch/ReadVariableOp:value:0/Bottleneck2/depthbatch/ReadVariableOp_1:value:0>Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp:value:0@Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( 
Bottleneck2/activation_4/ReluRelu+Bottleneck2/depthbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ0І
*Bottleneck2/projconv/Conv2D/ReadVariableOpReadVariableOp3bottleneck2_projconv_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0ш
Bottleneck2/projconv/Conv2DConv2D+Bottleneck2/activation_4/Relu:activations:02Bottleneck2/projconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

+Bottleneck2/projconv/BiasAdd/ReadVariableOpReadVariableOp4bottleneck2_projconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0М
Bottleneck2/projconv/BiasAddBiasAdd$Bottleneck2/projconv/Conv2D:output:03Bottleneck2/projconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
$Bottleneck2/projbatch/ReadVariableOpReadVariableOp-bottleneck2_projbatch_readvariableop_resource*
_output_shapes
:*
dtype0
&Bottleneck2/projbatch/ReadVariableOp_1ReadVariableOp/bottleneck2_projbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0А
5Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp>bottleneck2_projbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Д
7Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@bottleneck2_projbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0У
&Bottleneck2/projbatch/FusedBatchNormV3FusedBatchNormV3%Bottleneck2/projconv/BiasAdd:output:0,Bottleneck2/projbatch/ReadVariableOp:value:0.Bottleneck2/projbatch/ReadVariableOp_1:value:0=Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp:value:0?Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 
%Bottleneck2/shortbatch/ReadVariableOpReadVariableOp.bottleneck2_shortbatch_readvariableop_resource*
_output_shapes
:*
dtype0
'Bottleneck2/shortbatch/ReadVariableOp_1ReadVariableOp0bottleneck2_shortbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0В
6Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp?bottleneck2_shortbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ж
8Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbottleneck2_shortbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0О
'Bottleneck2/shortbatch/FusedBatchNormV3FusedBatchNormV3dropout_7/Identity:output:0-Bottleneck2/shortbatch/ReadVariableOp:value:0/Bottleneck2/shortbatch/ReadVariableOp_1:value:0>Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp:value:0@Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ:::::*
epsilon%o:*
is_training( Б
Bottleneck2/add_1/addAddV2*Bottleneck2/projbatch/FusedBatchNormV3:y:0+Bottleneck2/shortbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџz
Bottleneck2/activation_5/ReluReluBottleneck2/add_1/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџМ
max_pooling2d_8/MaxPoolMaxPool+Bottleneck2/activation_5/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
z
dropout_8/IdentityIdentity max_pooling2d_8/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  
flatten_2/ReshapeReshapedropout_8/Identity:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:		*
dtype0
dense_2/MatMulMatMulflatten_2/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџf
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџв
NoOpNoOp7^Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp9^Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_1&^Bottleneck1/depthbatch/ReadVariableOp(^Bottleneck1/depthbatch/ReadVariableOp_1-^Bottleneck1/depthconv/BiasAdd/ReadVariableOp/^Bottleneck1/depthconv/depthwise/ReadVariableOp5^Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp7^Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_1$^Bottleneck1/expbatch/ReadVariableOp&^Bottleneck1/expbatch/ReadVariableOp_1+^Bottleneck1/expconv/BiasAdd/ReadVariableOp*^Bottleneck1/expconv/Conv2D/ReadVariableOp6^Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp8^Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_1%^Bottleneck1/projbatch/ReadVariableOp'^Bottleneck1/projbatch/ReadVariableOp_1,^Bottleneck1/projconv/BiasAdd/ReadVariableOp+^Bottleneck1/projconv/Conv2D/ReadVariableOp7^Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp9^Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_1&^Bottleneck1/shortbatch/ReadVariableOp(^Bottleneck1/shortbatch/ReadVariableOp_17^Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp9^Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_1&^Bottleneck2/depthbatch/ReadVariableOp(^Bottleneck2/depthbatch/ReadVariableOp_1-^Bottleneck2/depthconv/BiasAdd/ReadVariableOp/^Bottleneck2/depthconv/depthwise/ReadVariableOp5^Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp7^Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_1$^Bottleneck2/expbatch/ReadVariableOp&^Bottleneck2/expbatch/ReadVariableOp_1+^Bottleneck2/expconv/BiasAdd/ReadVariableOp*^Bottleneck2/expconv/Conv2D/ReadVariableOp6^Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp8^Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_1%^Bottleneck2/projbatch/ReadVariableOp'^Bottleneck2/projbatch/ReadVariableOp_1,^Bottleneck2/projconv/BiasAdd/ReadVariableOp+^Bottleneck2/projconv/Conv2D/ReadVariableOp7^Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp9^Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_1&^Bottleneck2/shortbatch/ReadVariableOp(^Bottleneck2/shortbatch/ReadVariableOp_17^batch_normalization_34/FusedBatchNormV3/ReadVariableOp9^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_34/ReadVariableOp(^batch_normalization_34/ReadVariableOp_1!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџрр: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp6Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp2t
8Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_18Bottleneck1/depthbatch/FusedBatchNormV3/ReadVariableOp_12N
%Bottleneck1/depthbatch/ReadVariableOp%Bottleneck1/depthbatch/ReadVariableOp2R
'Bottleneck1/depthbatch/ReadVariableOp_1'Bottleneck1/depthbatch/ReadVariableOp_12\
,Bottleneck1/depthconv/BiasAdd/ReadVariableOp,Bottleneck1/depthconv/BiasAdd/ReadVariableOp2`
.Bottleneck1/depthconv/depthwise/ReadVariableOp.Bottleneck1/depthconv/depthwise/ReadVariableOp2l
4Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp4Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp2p
6Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_16Bottleneck1/expbatch/FusedBatchNormV3/ReadVariableOp_12J
#Bottleneck1/expbatch/ReadVariableOp#Bottleneck1/expbatch/ReadVariableOp2N
%Bottleneck1/expbatch/ReadVariableOp_1%Bottleneck1/expbatch/ReadVariableOp_12X
*Bottleneck1/expconv/BiasAdd/ReadVariableOp*Bottleneck1/expconv/BiasAdd/ReadVariableOp2V
)Bottleneck1/expconv/Conv2D/ReadVariableOp)Bottleneck1/expconv/Conv2D/ReadVariableOp2n
5Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp5Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp2r
7Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_17Bottleneck1/projbatch/FusedBatchNormV3/ReadVariableOp_12L
$Bottleneck1/projbatch/ReadVariableOp$Bottleneck1/projbatch/ReadVariableOp2P
&Bottleneck1/projbatch/ReadVariableOp_1&Bottleneck1/projbatch/ReadVariableOp_12Z
+Bottleneck1/projconv/BiasAdd/ReadVariableOp+Bottleneck1/projconv/BiasAdd/ReadVariableOp2X
*Bottleneck1/projconv/Conv2D/ReadVariableOp*Bottleneck1/projconv/Conv2D/ReadVariableOp2p
6Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp6Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp2t
8Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_18Bottleneck1/shortbatch/FusedBatchNormV3/ReadVariableOp_12N
%Bottleneck1/shortbatch/ReadVariableOp%Bottleneck1/shortbatch/ReadVariableOp2R
'Bottleneck1/shortbatch/ReadVariableOp_1'Bottleneck1/shortbatch/ReadVariableOp_12p
6Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp6Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp2t
8Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_18Bottleneck2/depthbatch/FusedBatchNormV3/ReadVariableOp_12N
%Bottleneck2/depthbatch/ReadVariableOp%Bottleneck2/depthbatch/ReadVariableOp2R
'Bottleneck2/depthbatch/ReadVariableOp_1'Bottleneck2/depthbatch/ReadVariableOp_12\
,Bottleneck2/depthconv/BiasAdd/ReadVariableOp,Bottleneck2/depthconv/BiasAdd/ReadVariableOp2`
.Bottleneck2/depthconv/depthwise/ReadVariableOp.Bottleneck2/depthconv/depthwise/ReadVariableOp2l
4Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp4Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp2p
6Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_16Bottleneck2/expbatch/FusedBatchNormV3/ReadVariableOp_12J
#Bottleneck2/expbatch/ReadVariableOp#Bottleneck2/expbatch/ReadVariableOp2N
%Bottleneck2/expbatch/ReadVariableOp_1%Bottleneck2/expbatch/ReadVariableOp_12X
*Bottleneck2/expconv/BiasAdd/ReadVariableOp*Bottleneck2/expconv/BiasAdd/ReadVariableOp2V
)Bottleneck2/expconv/Conv2D/ReadVariableOp)Bottleneck2/expconv/Conv2D/ReadVariableOp2n
5Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp5Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp2r
7Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_17Bottleneck2/projbatch/FusedBatchNormV3/ReadVariableOp_12L
$Bottleneck2/projbatch/ReadVariableOp$Bottleneck2/projbatch/ReadVariableOp2P
&Bottleneck2/projbatch/ReadVariableOp_1&Bottleneck2/projbatch/ReadVariableOp_12Z
+Bottleneck2/projconv/BiasAdd/ReadVariableOp+Bottleneck2/projconv/BiasAdd/ReadVariableOp2X
*Bottleneck2/projconv/Conv2D/ReadVariableOp*Bottleneck2/projconv/Conv2D/ReadVariableOp2p
6Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp6Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp2t
8Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_18Bottleneck2/shortbatch/FusedBatchNormV3/ReadVariableOp_12N
%Bottleneck2/shortbatch/ReadVariableOp%Bottleneck2/shortbatch/ReadVariableOp2R
'Bottleneck2/shortbatch/ReadVariableOp_1'Bottleneck2/shortbatch/ReadVariableOp_12p
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp6batch_normalization_34/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_18batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_34/ReadVariableOp%batch_normalization_34/ReadVariableOp2R
'batch_normalization_34/ReadVariableOp_1'batch_normalization_34/ReadVariableOp_12D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџрр
 
_user_specified_nameinputs
О

C__inference_expbatch_layer_call_and_return_conditional_losses_47549

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
Р

E__inference_depthbatch_layer_call_and_return_conditional_losses_51327

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
Ж
K
/__inference_max_pooling2d_7_layer_call_fn_50607

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_47524
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_48127

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџc

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
О

C__inference_expbatch_layer_call_and_return_conditional_losses_47281

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
љ
Г
D__inference_projbatch_layer_call_and_return_conditional_losses_47708

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
С

'__inference_dense_2_layer_call_fn_50962

inputs
unknown:		
	unknown_0:
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_48148o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs

Р
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_50289

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Р

E__inference_depthbatch_layer_call_and_return_conditional_losses_51079

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њ
Д
E__inference_depthbatch_layer_call_and_return_conditional_losses_51345

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs


є
B__inference_dense_2_layer_call_and_return_conditional_losses_48148

inputs1
matmul_readvariableop_resource:		-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:		*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
Р

E__inference_depthbatch_layer_call_and_return_conditional_losses_47613

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0:0:0:0:0:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 
_user_specified_nameinputs
цo
њ
F__inference_Bottleneck1_layer_call_and_return_conditional_losses_50602
x@
&expconv_conv2d_readvariableop_resource:5
'expconv_biasadd_readvariableop_resource:.
 expbatch_readvariableop_resource:0
"expbatch_readvariableop_1_resource:?
1expbatch_fusedbatchnormv3_readvariableop_resource:A
3expbatch_fusedbatchnormv3_readvariableop_1_resource:E
+depthconv_depthwise_readvariableop_resource:7
)depthconv_biasadd_readvariableop_resource:0
"depthbatch_readvariableop_resource:2
$depthbatch_readvariableop_1_resource:A
3depthbatch_fusedbatchnormv3_readvariableop_resource:C
5depthbatch_fusedbatchnormv3_readvariableop_1_resource:A
'projconv_conv2d_readvariableop_resource:6
(projconv_biasadd_readvariableop_resource:/
!projbatch_readvariableop_resource:1
#projbatch_readvariableop_1_resource:@
2projbatch_fusedbatchnormv3_readvariableop_resource:B
4projbatch_fusedbatchnormv3_readvariableop_1_resource:0
"shortbatch_readvariableop_resource:2
$shortbatch_readvariableop_1_resource:A
3shortbatch_fusedbatchnormv3_readvariableop_resource:C
5shortbatch_fusedbatchnormv3_readvariableop_1_resource:
identityЂdepthbatch/AssignNewValueЂdepthbatch/AssignNewValue_1Ђ*depthbatch/FusedBatchNormV3/ReadVariableOpЂ,depthbatch/FusedBatchNormV3/ReadVariableOp_1Ђdepthbatch/ReadVariableOpЂdepthbatch/ReadVariableOp_1Ђ depthconv/BiasAdd/ReadVariableOpЂ"depthconv/depthwise/ReadVariableOpЂexpbatch/AssignNewValueЂexpbatch/AssignNewValue_1Ђ(expbatch/FusedBatchNormV3/ReadVariableOpЂ*expbatch/FusedBatchNormV3/ReadVariableOp_1Ђexpbatch/ReadVariableOpЂexpbatch/ReadVariableOp_1Ђexpconv/BiasAdd/ReadVariableOpЂexpconv/Conv2D/ReadVariableOpЂprojbatch/AssignNewValueЂprojbatch/AssignNewValue_1Ђ)projbatch/FusedBatchNormV3/ReadVariableOpЂ+projbatch/FusedBatchNormV3/ReadVariableOp_1Ђprojbatch/ReadVariableOpЂprojbatch/ReadVariableOp_1Ђprojconv/BiasAdd/ReadVariableOpЂprojconv/Conv2D/ReadVariableOpЂshortbatch/AssignNewValueЂshortbatch/AssignNewValue_1Ђ*shortbatch/FusedBatchNormV3/ReadVariableOpЂ,shortbatch/FusedBatchNormV3/ReadVariableOp_1Ђshortbatch/ReadVariableOpЂshortbatch/ReadVariableOp_1
expconv/Conv2D/ReadVariableOpReadVariableOp&expconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Є
expconv/Conv2DConv2Dx%expconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

expconv/BiasAdd/ReadVariableOpReadVariableOp'expconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
expconv/BiasAddBiasAddexpconv/Conv2D:output:0&expconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88t
expbatch/ReadVariableOpReadVariableOp expbatch_readvariableop_resource*
_output_shapes
:*
dtype0x
expbatch/ReadVariableOp_1ReadVariableOp"expbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
(expbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp1expbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
*expbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3expbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
expbatch/FusedBatchNormV3FusedBatchNormV3expconv/BiasAdd:output:0expbatch/ReadVariableOp:value:0!expbatch/ReadVariableOp_1:value:00expbatch/FusedBatchNormV3/ReadVariableOp:value:02expbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
exponential_avg_factor%
з#<ъ
expbatch/AssignNewValueAssignVariableOp1expbatch_fusedbatchnormv3_readvariableop_resource&expbatch/FusedBatchNormV3:batch_mean:0)^expbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(є
expbatch/AssignNewValue_1AssignVariableOp3expbatch_fusedbatchnormv3_readvariableop_1_resource*expbatch/FusedBatchNormV3:batch_variance:0+^expbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(p
activation/ReluReluexpbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88
"depthconv/depthwise/ReadVariableOpReadVariableOp+depthconv_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0r
depthconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            r
!depthconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      й
depthconv/depthwiseDepthwiseConv2dNativeactivation/Relu:activations:0*depthconv/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

 depthconv/BiasAdd/ReadVariableOpReadVariableOp)depthconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
depthconv/BiasAddBiasAdddepthconv/depthwise:output:0(depthconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88x
depthbatch/ReadVariableOpReadVariableOp"depthbatch_readvariableop_resource*
_output_shapes
:*
dtype0|
depthbatch/ReadVariableOp_1ReadVariableOp$depthbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
*depthbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp3depthbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
,depthbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5depthbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
depthbatch/FusedBatchNormV3FusedBatchNormV3depthconv/BiasAdd:output:0!depthbatch/ReadVariableOp:value:0#depthbatch/ReadVariableOp_1:value:02depthbatch/FusedBatchNormV3/ReadVariableOp:value:04depthbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
exponential_avg_factor%
з#<ђ
depthbatch/AssignNewValueAssignVariableOp3depthbatch_fusedbatchnormv3_readvariableop_resource(depthbatch/FusedBatchNormV3:batch_mean:0+^depthbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ќ
depthbatch/AssignNewValue_1AssignVariableOp5depthbatch_fusedbatchnormv3_readvariableop_1_resource,depthbatch/FusedBatchNormV3:batch_variance:0-^depthbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(t
activation_1/ReluReludepthbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88
projconv/Conv2D/ReadVariableOpReadVariableOp'projconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
projconv/Conv2DConv2Dactivation_1/Relu:activations:0&projconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

projconv/BiasAdd/ReadVariableOpReadVariableOp(projconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
projconv/BiasAddBiasAddprojconv/Conv2D:output:0'projconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88v
projbatch/ReadVariableOpReadVariableOp!projbatch_readvariableop_resource*
_output_shapes
:*
dtype0z
projbatch/ReadVariableOp_1ReadVariableOp#projbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
)projbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp2projbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
+projbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4projbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
projbatch/FusedBatchNormV3FusedBatchNormV3projconv/BiasAdd:output:0 projbatch/ReadVariableOp:value:0"projbatch/ReadVariableOp_1:value:01projbatch/FusedBatchNormV3/ReadVariableOp:value:03projbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
exponential_avg_factor%
з#<ю
projbatch/AssignNewValueAssignVariableOp2projbatch_fusedbatchnormv3_readvariableop_resource'projbatch/FusedBatchNormV3:batch_mean:0*^projbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ј
projbatch/AssignNewValue_1AssignVariableOp4projbatch_fusedbatchnormv3_readvariableop_1_resource+projbatch/FusedBatchNormV3:batch_variance:0,^projbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(x
shortbatch/ReadVariableOpReadVariableOp"shortbatch_readvariableop_resource*
_output_shapes
:*
dtype0|
shortbatch/ReadVariableOp_1ReadVariableOp$shortbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
*shortbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp3shortbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
,shortbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5shortbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0і
shortbatch/FusedBatchNormV3FusedBatchNormV3x!shortbatch/ReadVariableOp:value:0#shortbatch/ReadVariableOp_1:value:02shortbatch/FusedBatchNormV3/ReadVariableOp:value:04shortbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
exponential_avg_factor%
з#<ђ
shortbatch/AssignNewValueAssignVariableOp3shortbatch_fusedbatchnormv3_readvariableop_resource(shortbatch/FusedBatchNormV3:batch_mean:0+^shortbatch/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ќ
shortbatch/AssignNewValue_1AssignVariableOp5shortbatch_fusedbatchnormv3_readvariableop_1_resource,shortbatch/FusedBatchNormV3:batch_variance:0-^shortbatch/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
add/addAddV2projbatch/FusedBatchNormV3:y:0shortbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88`
activation_2/ReluReluadd/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ88v
IdentityIdentityactivation_2/Relu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ88Р
NoOpNoOp^depthbatch/AssignNewValue^depthbatch/AssignNewValue_1+^depthbatch/FusedBatchNormV3/ReadVariableOp-^depthbatch/FusedBatchNormV3/ReadVariableOp_1^depthbatch/ReadVariableOp^depthbatch/ReadVariableOp_1!^depthconv/BiasAdd/ReadVariableOp#^depthconv/depthwise/ReadVariableOp^expbatch/AssignNewValue^expbatch/AssignNewValue_1)^expbatch/FusedBatchNormV3/ReadVariableOp+^expbatch/FusedBatchNormV3/ReadVariableOp_1^expbatch/ReadVariableOp^expbatch/ReadVariableOp_1^expconv/BiasAdd/ReadVariableOp^expconv/Conv2D/ReadVariableOp^projbatch/AssignNewValue^projbatch/AssignNewValue_1*^projbatch/FusedBatchNormV3/ReadVariableOp,^projbatch/FusedBatchNormV3/ReadVariableOp_1^projbatch/ReadVariableOp^projbatch/ReadVariableOp_1 ^projconv/BiasAdd/ReadVariableOp^projconv/Conv2D/ReadVariableOp^shortbatch/AssignNewValue^shortbatch/AssignNewValue_1+^shortbatch/FusedBatchNormV3/ReadVariableOp-^shortbatch/FusedBatchNormV3/ReadVariableOp_1^shortbatch/ReadVariableOp^shortbatch/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:џџџџџџџџџ88: : : : : : : : : : : : : : : : : : : : : : 26
depthbatch/AssignNewValuedepthbatch/AssignNewValue2:
depthbatch/AssignNewValue_1depthbatch/AssignNewValue_12X
*depthbatch/FusedBatchNormV3/ReadVariableOp*depthbatch/FusedBatchNormV3/ReadVariableOp2\
,depthbatch/FusedBatchNormV3/ReadVariableOp_1,depthbatch/FusedBatchNormV3/ReadVariableOp_126
depthbatch/ReadVariableOpdepthbatch/ReadVariableOp2:
depthbatch/ReadVariableOp_1depthbatch/ReadVariableOp_12D
 depthconv/BiasAdd/ReadVariableOp depthconv/BiasAdd/ReadVariableOp2H
"depthconv/depthwise/ReadVariableOp"depthconv/depthwise/ReadVariableOp22
expbatch/AssignNewValueexpbatch/AssignNewValue26
expbatch/AssignNewValue_1expbatch/AssignNewValue_12T
(expbatch/FusedBatchNormV3/ReadVariableOp(expbatch/FusedBatchNormV3/ReadVariableOp2X
*expbatch/FusedBatchNormV3/ReadVariableOp_1*expbatch/FusedBatchNormV3/ReadVariableOp_122
expbatch/ReadVariableOpexpbatch/ReadVariableOp26
expbatch/ReadVariableOp_1expbatch/ReadVariableOp_12@
expconv/BiasAdd/ReadVariableOpexpconv/BiasAdd/ReadVariableOp2>
expconv/Conv2D/ReadVariableOpexpconv/Conv2D/ReadVariableOp24
projbatch/AssignNewValueprojbatch/AssignNewValue28
projbatch/AssignNewValue_1projbatch/AssignNewValue_12V
)projbatch/FusedBatchNormV3/ReadVariableOp)projbatch/FusedBatchNormV3/ReadVariableOp2Z
+projbatch/FusedBatchNormV3/ReadVariableOp_1+projbatch/FusedBatchNormV3/ReadVariableOp_124
projbatch/ReadVariableOpprojbatch/ReadVariableOp28
projbatch/ReadVariableOp_1projbatch/ReadVariableOp_12B
projconv/BiasAdd/ReadVariableOpprojconv/BiasAdd/ReadVariableOp2@
projconv/Conv2D/ReadVariableOpprojconv/Conv2D/ReadVariableOp26
shortbatch/AssignNewValueshortbatch/AssignNewValue2:
shortbatch/AssignNewValue_1shortbatch/AssignNewValue_12X
*shortbatch/FusedBatchNormV3/ReadVariableOp*shortbatch/FusedBatchNormV3/ReadVariableOp2\
,shortbatch/FusedBatchNormV3/ReadVariableOp_1,shortbatch/FusedBatchNormV3/ReadVariableOp_126
shortbatch/ReadVariableOpshortbatch/ReadVariableOp2:
shortbatch/ReadVariableOp_1shortbatch/ReadVariableOp_1:R N
/
_output_shapes
:џџџџџџџџџ88

_user_specified_nameX
К
є
'__inference_model_2_layer_call_fn_49764

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:$

unknown_27:0

unknown_28:0

unknown_29:0

unknown_30:0

unknown_31:0

unknown_32:0$

unknown_33:0

unknown_34:0

unknown_35:0

unknown_36:0

unknown_37:0

unknown_38:0$

unknown_39:0

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:		

unknown_50:
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*D
_read_only_resource_inputs&
$"	
 #$%&)*+,/034*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_48969o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџрр: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџрр
 
_user_specified_nameinputs
Ж
K
/__inference_max_pooling2d_8_layer_call_fn_50910

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_47792
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї
Х
*__inference_depthbatch_layer_call_fn_51061

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_depthbatch_layer_call_and_return_conditional_losses_47376
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
юX

F__inference_Bottleneck1_layer_call_and_return_conditional_losses_47937
x@
&expconv_conv2d_readvariableop_resource:5
'expconv_biasadd_readvariableop_resource:.
 expbatch_readvariableop_resource:0
"expbatch_readvariableop_1_resource:?
1expbatch_fusedbatchnormv3_readvariableop_resource:A
3expbatch_fusedbatchnormv3_readvariableop_1_resource:E
+depthconv_depthwise_readvariableop_resource:7
)depthconv_biasadd_readvariableop_resource:0
"depthbatch_readvariableop_resource:2
$depthbatch_readvariableop_1_resource:A
3depthbatch_fusedbatchnormv3_readvariableop_resource:C
5depthbatch_fusedbatchnormv3_readvariableop_1_resource:A
'projconv_conv2d_readvariableop_resource:6
(projconv_biasadd_readvariableop_resource:/
!projbatch_readvariableop_resource:1
#projbatch_readvariableop_1_resource:@
2projbatch_fusedbatchnormv3_readvariableop_resource:B
4projbatch_fusedbatchnormv3_readvariableop_1_resource:0
"shortbatch_readvariableop_resource:2
$shortbatch_readvariableop_1_resource:A
3shortbatch_fusedbatchnormv3_readvariableop_resource:C
5shortbatch_fusedbatchnormv3_readvariableop_1_resource:
identityЂ*depthbatch/FusedBatchNormV3/ReadVariableOpЂ,depthbatch/FusedBatchNormV3/ReadVariableOp_1Ђdepthbatch/ReadVariableOpЂdepthbatch/ReadVariableOp_1Ђ depthconv/BiasAdd/ReadVariableOpЂ"depthconv/depthwise/ReadVariableOpЂ(expbatch/FusedBatchNormV3/ReadVariableOpЂ*expbatch/FusedBatchNormV3/ReadVariableOp_1Ђexpbatch/ReadVariableOpЂexpbatch/ReadVariableOp_1Ђexpconv/BiasAdd/ReadVariableOpЂexpconv/Conv2D/ReadVariableOpЂ)projbatch/FusedBatchNormV3/ReadVariableOpЂ+projbatch/FusedBatchNormV3/ReadVariableOp_1Ђprojbatch/ReadVariableOpЂprojbatch/ReadVariableOp_1Ђprojconv/BiasAdd/ReadVariableOpЂprojconv/Conv2D/ReadVariableOpЂ*shortbatch/FusedBatchNormV3/ReadVariableOpЂ,shortbatch/FusedBatchNormV3/ReadVariableOp_1Ђshortbatch/ReadVariableOpЂshortbatch/ReadVariableOp_1
expconv/Conv2D/ReadVariableOpReadVariableOp&expconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Є
expconv/Conv2DConv2Dx%expconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

expconv/BiasAdd/ReadVariableOpReadVariableOp'expconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
expconv/BiasAddBiasAddexpconv/Conv2D:output:0&expconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88t
expbatch/ReadVariableOpReadVariableOp expbatch_readvariableop_resource*
_output_shapes
:*
dtype0x
expbatch/ReadVariableOp_1ReadVariableOp"expbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
(expbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp1expbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
*expbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3expbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ѕ
expbatch/FusedBatchNormV3FusedBatchNormV3expconv/BiasAdd:output:0expbatch/ReadVariableOp:value:0!expbatch/ReadVariableOp_1:value:00expbatch/FusedBatchNormV3/ReadVariableOp:value:02expbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
is_training( p
activation/ReluReluexpbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88
"depthconv/depthwise/ReadVariableOpReadVariableOp+depthconv_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0r
depthconv/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            r
!depthconv/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      й
depthconv/depthwiseDepthwiseConv2dNativeactivation/Relu:activations:0*depthconv/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

 depthconv/BiasAdd/ReadVariableOpReadVariableOp)depthconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
depthconv/BiasAddBiasAdddepthconv/depthwise:output:0(depthconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88x
depthbatch/ReadVariableOpReadVariableOp"depthbatch_readvariableop_resource*
_output_shapes
:*
dtype0|
depthbatch/ReadVariableOp_1ReadVariableOp$depthbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
*depthbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp3depthbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
,depthbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5depthbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
depthbatch/FusedBatchNormV3FusedBatchNormV3depthconv/BiasAdd:output:0!depthbatch/ReadVariableOp:value:0#depthbatch/ReadVariableOp_1:value:02depthbatch/FusedBatchNormV3/ReadVariableOp:value:04depthbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
is_training( t
activation_1/ReluReludepthbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88
projconv/Conv2D/ReadVariableOpReadVariableOp'projconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
projconv/Conv2DConv2Dactivation_1/Relu:activations:0&projconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
paddingSAME*
strides

projconv/BiasAdd/ReadVariableOpReadVariableOp(projconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
projconv/BiasAddBiasAddprojconv/Conv2D:output:0'projconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ88v
projbatch/ReadVariableOpReadVariableOp!projbatch_readvariableop_resource*
_output_shapes
:*
dtype0z
projbatch/ReadVariableOp_1ReadVariableOp#projbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
)projbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp2projbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
+projbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4projbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ћ
projbatch/FusedBatchNormV3FusedBatchNormV3projconv/BiasAdd:output:0 projbatch/ReadVariableOp:value:0"projbatch/ReadVariableOp_1:value:01projbatch/FusedBatchNormV3/ReadVariableOp:value:03projbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
is_training( x
shortbatch/ReadVariableOpReadVariableOp"shortbatch_readvariableop_resource*
_output_shapes
:*
dtype0|
shortbatch/ReadVariableOp_1ReadVariableOp$shortbatch_readvariableop_1_resource*
_output_shapes
:*
dtype0
*shortbatch/FusedBatchNormV3/ReadVariableOpReadVariableOp3shortbatch_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
,shortbatch/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5shortbatch_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ш
shortbatch/FusedBatchNormV3FusedBatchNormV3x!shortbatch/ReadVariableOp:value:0#shortbatch/ReadVariableOp_1:value:02shortbatch/FusedBatchNormV3/ReadVariableOp:value:04shortbatch/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ88:::::*
epsilon%o:*
is_training( 
add/addAddV2projbatch/FusedBatchNormV3:y:0shortbatch/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88`
activation_2/ReluReluadd/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ88v
IdentityIdentityactivation_2/Relu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ88о
NoOpNoOp+^depthbatch/FusedBatchNormV3/ReadVariableOp-^depthbatch/FusedBatchNormV3/ReadVariableOp_1^depthbatch/ReadVariableOp^depthbatch/ReadVariableOp_1!^depthconv/BiasAdd/ReadVariableOp#^depthconv/depthwise/ReadVariableOp)^expbatch/FusedBatchNormV3/ReadVariableOp+^expbatch/FusedBatchNormV3/ReadVariableOp_1^expbatch/ReadVariableOp^expbatch/ReadVariableOp_1^expconv/BiasAdd/ReadVariableOp^expconv/Conv2D/ReadVariableOp*^projbatch/FusedBatchNormV3/ReadVariableOp,^projbatch/FusedBatchNormV3/ReadVariableOp_1^projbatch/ReadVariableOp^projbatch/ReadVariableOp_1 ^projconv/BiasAdd/ReadVariableOp^projconv/Conv2D/ReadVariableOp+^shortbatch/FusedBatchNormV3/ReadVariableOp-^shortbatch/FusedBatchNormV3/ReadVariableOp_1^shortbatch/ReadVariableOp^shortbatch/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:џџџџџџџџџ88: : : : : : : : : : : : : : : : : : : : : : 2X
*depthbatch/FusedBatchNormV3/ReadVariableOp*depthbatch/FusedBatchNormV3/ReadVariableOp2\
,depthbatch/FusedBatchNormV3/ReadVariableOp_1,depthbatch/FusedBatchNormV3/ReadVariableOp_126
depthbatch/ReadVariableOpdepthbatch/ReadVariableOp2:
depthbatch/ReadVariableOp_1depthbatch/ReadVariableOp_12D
 depthconv/BiasAdd/ReadVariableOp depthconv/BiasAdd/ReadVariableOp2H
"depthconv/depthwise/ReadVariableOp"depthconv/depthwise/ReadVariableOp2T
(expbatch/FusedBatchNormV3/ReadVariableOp(expbatch/FusedBatchNormV3/ReadVariableOp2X
*expbatch/FusedBatchNormV3/ReadVariableOp_1*expbatch/FusedBatchNormV3/ReadVariableOp_122
expbatch/ReadVariableOpexpbatch/ReadVariableOp26
expbatch/ReadVariableOp_1expbatch/ReadVariableOp_12@
expconv/BiasAdd/ReadVariableOpexpconv/BiasAdd/ReadVariableOp2>
expconv/Conv2D/ReadVariableOpexpconv/Conv2D/ReadVariableOp2V
)projbatch/FusedBatchNormV3/ReadVariableOp)projbatch/FusedBatchNormV3/ReadVariableOp2Z
+projbatch/FusedBatchNormV3/ReadVariableOp_1+projbatch/FusedBatchNormV3/ReadVariableOp_124
projbatch/ReadVariableOpprojbatch/ReadVariableOp28
projbatch/ReadVariableOp_1projbatch/ReadVariableOp_12B
projconv/BiasAdd/ReadVariableOpprojconv/BiasAdd/ReadVariableOp2@
projconv/Conv2D/ReadVariableOpprojconv/Conv2D/ReadVariableOp2X
*shortbatch/FusedBatchNormV3/ReadVariableOp*shortbatch/FusedBatchNormV3/ReadVariableOp2\
,shortbatch/FusedBatchNormV3/ReadVariableOp_1,shortbatch/FusedBatchNormV3/ReadVariableOp_126
shortbatch/ReadVariableOpshortbatch/ReadVariableOp2:
shortbatch/ReadVariableOp_1shortbatch/ReadVariableOp_1:R N
/
_output_shapes
:џџџџџџџџџ88

_user_specified_nameX
О

c
D__inference_dropout_8_layer_call_and_return_conditional_losses_48298

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџq
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџa
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
њ
Д
E__inference_depthbatch_layer_call_and_return_conditional_losses_51097

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
П

D__inference_projbatch_layer_call_and_return_conditional_losses_51389

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Н
E
)__inference_dropout_6_layer_call_fn_50314

inputs
identityЗ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ88* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_47851h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ88"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ88:W S
/
_output_shapes
:џџџџџџџџџ88
 
_user_specified_nameinputs
О

C__inference_expbatch_layer_call_and_return_conditional_losses_51017

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
О

c
D__inference_dropout_6_layer_call_and_return_conditional_losses_50336

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ88C
dropout/ShapeShapeinputs*
T0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ88w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ88q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ88"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ88:W S
/
_output_shapes
:џџџџџџџџџ88
 
_user_specified_nameinputs
­
В
+__inference_Bottleneck1_layer_call_fn_50385
x!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ88*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Bottleneck1_layer_call_and_return_conditional_losses_47937w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ88`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:џџџџџџџџџ88: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:џџџџџџџџџ88

_user_specified_nameX
О

c
D__inference_dropout_6_layer_call_and_return_conditional_losses_48710

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ88C
dropout/ShapeShapeinputs*
T0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ88*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ88w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ88q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ88a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ88"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ88:W S
/
_output_shapes
:џџџџџџџџџ88
 
_user_specified_nameinputs
Щ
G
+__inference_rescaling_2_layer_call_fn_50200

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџрр* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_rescaling_2_layer_call_and_return_conditional_losses_47811j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџрр"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџрр:Y U
1
_output_shapes
:џџџџџџџџџрр
 
_user_specified_nameinputs"Е	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Д
serving_default 
E
input_3:
serving_default_input_3:0џџџџџџџџџрр;
dense_20
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:іУ
Л
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer-10
layer_with_weights-3
layer-11
layer-12
layer-13
layer-14
layer_with_weights-4
layer-15
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
(
	keras_api"
_tf_keras_layer
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer
н
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias
 )_jit_compiled_convolution_op"
_tf_keras_layer
ъ
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0axis
	1gamma
2beta
3moving_mean
4moving_variance"
_tf_keras_layer
Ѕ
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
М
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
G_random_generator"
_tf_keras_layer

H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Nexpconv
Oexpbatch
P	depthconv
Q
depthbatch
Rprojconv
S	projbatch
T
shortbatch"
_tf_keras_layer
Ѕ
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
М
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
a_random_generator"
_tf_keras_layer

b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
hexpconv
iexpbatch
j	depthconv
k
depthbatch
lprojconv
m	projbatch
n
shortbatch"
_tf_keras_layer
Ѕ
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer
М
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
{_random_generator"
_tf_keras_layer
Ї
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
У
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
ф
'0
(1
12
23
34
45
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
 28
Ё29
Ђ30
Ѓ31
Є32
Ѕ33
І34
Ї35
Ј36
Љ37
Њ38
Ћ39
Ќ40
­41
Ў42
Џ43
А44
Б45
В46
Г47
Д48
Е49
50
51"
trackable_list_wrapper
Ф
'0
(1
12
23
4
5
6
7
8
9
10
11
12
13
14
15
16
17
 18
Ё19
Ђ20
Ѓ21
Є22
Ѕ23
І24
Ї25
Ј26
Љ27
Њ28
Ћ29
Ќ30
­31
32
33"
trackable_list_wrapper
 "
trackable_list_wrapper
Я
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
й
Лtrace_0
Мtrace_1
Нtrace_2
Оtrace_32ц
'__inference_model_2_layer_call_fn_48262
'__inference_model_2_layer_call_fn_49655
'__inference_model_2_layer_call_fn_49764
'__inference_model_2_layer_call_fn_49185П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЛtrace_0zМtrace_1zНtrace_2zОtrace_3
Х
Пtrace_0
Рtrace_1
Сtrace_2
Тtrace_32в
B__inference_model_2_layer_call_and_return_conditional_losses_49969
B__inference_model_2_layer_call_and_return_conditional_losses_50195
B__inference_model_2_layer_call_and_return_conditional_losses_49308
B__inference_model_2_layer_call_and_return_conditional_losses_49431П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zПtrace_0zРtrace_1zСtrace_2zТtrace_3
ЫBШ
 __inference__wrapped_model_47183input_3"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј
	Уiter
Фbeta_1
Хbeta_2

Цdecay'm(m1m2m	m	m	m	m	m	m	m	m	m	m	m	m	m	m 	mЁ	mЂ	 mЃ	ЁmЄ	ЂmЅ	ЃmІ	ЄmЇ	ЅmЈ	ІmЉ	ЇmЊ	ЈmЋ	ЉmЌ	Њm­	ЋmЎ	ЌmЏ	­mА'vБ(vВ1vГ2vД	vЕ	vЖ	vЗ	vИ	vЙ	vК	vЛ	vМ	vН	vО	vП	vР	vС	vТ	vУ	vФ	 vХ	ЁvЦ	ЂvЧ	ЃvШ	ЄvЩ	ЅvЪ	ІvЫ	ЇvЬ	ЈvЭ	ЉvЮ	ЊvЯ	Ћvа	Ќvб	­vв"
	optimizer
-
Чserving_default"
signature_map
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
ё
Эtrace_02в
+__inference_rescaling_2_layer_call_fn_50200Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЭtrace_0

Юtrace_02э
F__inference_rescaling_2_layer_call_and_return_conditional_losses_50208Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЮtrace_0
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
я
дtrace_02а
)__inference_conv2d_18_layer_call_fn_50217Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zдtrace_0

еtrace_02ы
D__inference_conv2d_18_layer_call_and_return_conditional_losses_50227Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zеtrace_0
*:(2conv2d_18/kernel
:2conv2d_18/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
<
10
21
32
43"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
В
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
с
лtrace_0
мtrace_12І
6__inference_batch_normalization_34_layer_call_fn_50240
6__inference_batch_normalization_34_layer_call_fn_50253Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zлtrace_0zмtrace_1

нtrace_0
оtrace_12м
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_50271
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_50289Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zнtrace_0zоtrace_1
 "
trackable_list_wrapper
*:(2batch_normalization_34/gamma
):'2batch_normalization_34/beta
2:0 (2"batch_normalization_34/moving_mean
6:4 (2&batch_normalization_34/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
ђ
фtrace_02г
,__inference_activation_2_layer_call_fn_50294Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zфtrace_0

хtrace_02ю
G__inference_activation_2_layer_call_and_return_conditional_losses_50299Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zхtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
ѕ
ыtrace_02ж
/__inference_max_pooling2d_6_layer_call_fn_50304Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zыtrace_0

ьtrace_02ё
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_50309Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zьtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
эnon_trainable_variables
юlayers
яmetrics
 №layer_regularization_losses
ёlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
Ч
ђtrace_0
ѓtrace_12
)__inference_dropout_6_layer_call_fn_50314
)__inference_dropout_6_layer_call_fn_50319Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zђtrace_0zѓtrace_1
§
єtrace_0
ѕtrace_12Т
D__inference_dropout_6_layer_call_and_return_conditional_losses_50324
D__inference_dropout_6_layer_call_and_return_conditional_losses_50336Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zєtrace_0zѕtrace_1
"
_generic_user_object
м
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21"
trackable_list_wrapper

0
1
2
3
4
5
6
7
8
9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
В
іnon_trainable_variables
їlayers
јmetrics
 љlayer_regularization_losses
њlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
в
ћtrace_0
ќtrace_12
+__inference_Bottleneck1_layer_call_fn_50385
+__inference_Bottleneck1_layer_call_fn_50434К
БВ­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 zћtrace_0zќtrace_1

§trace_0
ўtrace_12Э
F__inference_Bottleneck1_layer_call_and_return_conditional_losses_50518
F__inference_Bottleneck1_layer_call_and_return_conditional_losses_50602К
БВ­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 z§trace_0zўtrace_1
ц
џ	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
moving_variance"
_tf_keras_layer
№
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
depthwise_kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
moving_variance"
_tf_keras_layer
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses
kernel
	bias
!Ё_jit_compiled_convolution_op"
_tf_keras_layer
ѕ
Ђ	variables
Ѓtrainable_variables
Єregularization_losses
Ѕ	keras_api
І__call__
+Ї&call_and_return_all_conditional_losses
	Јaxis

gamma
	beta
moving_mean
moving_variance"
_tf_keras_layer
ѕ
Љ	variables
Њtrainable_variables
Ћregularization_losses
Ќ	keras_api
­__call__
+Ў&call_and_return_all_conditional_losses
	Џaxis

gamma
	beta
moving_mean
moving_variance"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
ѕ
Еtrace_02ж
/__inference_max_pooling2d_7_layer_call_fn_50607Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЕtrace_0

Жtrace_02ё
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_50612Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЖtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
Ч
Мtrace_0
Нtrace_12
)__inference_dropout_7_layer_call_fn_50617
)__inference_dropout_7_layer_call_fn_50622Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zМtrace_0zНtrace_1
§
Оtrace_0
Пtrace_12Т
D__inference_dropout_7_layer_call_and_return_conditional_losses_50627
D__inference_dropout_7_layer_call_and_return_conditional_losses_50639Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zОtrace_0zПtrace_1
"
_generic_user_object
м
 0
Ё1
Ђ2
Ѓ3
Є4
Ѕ5
І6
Ї7
Ј8
Љ9
Њ10
Ћ11
Ќ12
­13
Ў14
Џ15
А16
Б17
В18
Г19
Д20
Е21"
trackable_list_wrapper

 0
Ё1
Ђ2
Ѓ3
Є4
Ѕ5
І6
Ї7
Ј8
Љ9
Њ10
Ћ11
Ќ12
­13"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
в
Хtrace_0
Цtrace_12
+__inference_Bottleneck2_layer_call_fn_50688
+__inference_Bottleneck2_layer_call_fn_50737К
БВ­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 zХtrace_0zЦtrace_1

Чtrace_0
Шtrace_12Э
F__inference_Bottleneck2_layer_call_and_return_conditional_losses_50821
F__inference_Bottleneck2_layer_call_and_return_conditional_losses_50905К
БВ­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 zЧtrace_0zШtrace_1
ц
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses
 kernel
	Ёbias
!Я_jit_compiled_convolution_op"
_tf_keras_layer
ѕ
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses
	жaxis

Ђgamma
	Ѓbeta
Ўmoving_mean
Џmoving_variance"
_tf_keras_layer
№
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
л__call__
+м&call_and_return_all_conditional_losses
Єdepthwise_kernel
	Ѕbias
!н_jit_compiled_convolution_op"
_tf_keras_layer
ѕ
о	variables
пtrainable_variables
рregularization_losses
с	keras_api
т__call__
+у&call_and_return_all_conditional_losses
	фaxis

Іgamma
	Їbeta
Аmoving_mean
Бmoving_variance"
_tf_keras_layer
ц
х	variables
цtrainable_variables
чregularization_losses
ш	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses
Јkernel
	Љbias
!ы_jit_compiled_convolution_op"
_tf_keras_layer
ѕ
ь	variables
эtrainable_variables
юregularization_losses
я	keras_api
№__call__
+ё&call_and_return_all_conditional_losses
	ђaxis

Њgamma
	Ћbeta
Вmoving_mean
Гmoving_variance"
_tf_keras_layer
ѕ
ѓ	variables
єtrainable_variables
ѕregularization_losses
і	keras_api
ї__call__
+ј&call_and_return_all_conditional_losses
	љaxis

Ќgamma
	­beta
Дmoving_mean
Еmoving_variance"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
њnon_trainable_variables
ћlayers
ќmetrics
 §layer_regularization_losses
ўlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
ѕ
џtrace_02ж
/__inference_max_pooling2d_8_layer_call_fn_50910Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zџtrace_0

trace_02ё
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_50915Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
Ч
trace_0
trace_12
)__inference_dropout_8_layer_call_fn_50920
)__inference_dropout_8_layer_call_fn_50925Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
§
trace_0
trace_12Т
D__inference_dropout_8_layer_call_and_return_conditional_losses_50930
D__inference_dropout_8_layer_call_and_return_conditional_losses_50942Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
я
trace_02а
)__inference_flatten_2_layer_call_fn_50947Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ы
D__inference_flatten_2_layer_call_and_return_conditional_losses_50953Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
э
trace_02Ю
'__inference_dense_2_layer_call_fn_50962Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02щ
B__inference_dense_2_layer_call_and_return_conditional_losses_50973Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
!:		2dense_2/kernel
:2dense_2/bias
4:22Bottleneck1/expconv/kernel
&:$2Bottleneck1/expconv/bias
(:&2Bottleneck1/expbatch/gamma
':%2Bottleneck1/expbatch/beta
@:>2&Bottleneck1/depthconv/depthwise_kernel
(:&2Bottleneck1/depthconv/bias
*:(2Bottleneck1/depthbatch/gamma
):'2Bottleneck1/depthbatch/beta
5:32Bottleneck1/projconv/kernel
':%2Bottleneck1/projconv/bias
):'2Bottleneck1/projbatch/gamma
(:&2Bottleneck1/projbatch/beta
*:(2Bottleneck1/shortbatch/gamma
):'2Bottleneck1/shortbatch/beta
0:. (2 Bottleneck1/expbatch/moving_mean
4:2 (2$Bottleneck1/expbatch/moving_variance
2:0 (2"Bottleneck1/depthbatch/moving_mean
6:4 (2&Bottleneck1/depthbatch/moving_variance
1:/ (2!Bottleneck1/projbatch/moving_mean
5:3 (2%Bottleneck1/projbatch/moving_variance
2:0 (2"Bottleneck1/shortbatch/moving_mean
6:4 (2&Bottleneck1/shortbatch/moving_variance
4:202Bottleneck2/expconv/kernel
&:$02Bottleneck2/expconv/bias
(:&02Bottleneck2/expbatch/gamma
':%02Bottleneck2/expbatch/beta
@:>02&Bottleneck2/depthconv/depthwise_kernel
(:&02Bottleneck2/depthconv/bias
*:(02Bottleneck2/depthbatch/gamma
):'02Bottleneck2/depthbatch/beta
5:302Bottleneck2/projconv/kernel
':%2Bottleneck2/projconv/bias
):'2Bottleneck2/projbatch/gamma
(:&2Bottleneck2/projbatch/beta
*:(2Bottleneck2/shortbatch/gamma
):'2Bottleneck2/shortbatch/beta
0:.0 (2 Bottleneck2/expbatch/moving_mean
4:20 (2$Bottleneck2/expbatch/moving_variance
2:00 (2"Bottleneck2/depthbatch/moving_mean
6:40 (2&Bottleneck2/depthbatch/moving_variance
1:/ (2!Bottleneck2/projbatch/moving_mean
5:3 (2%Bottleneck2/projbatch/moving_variance
2:0 (2"Bottleneck2/shortbatch/moving_mean
6:4 (2&Bottleneck2/shortbatch/moving_variance
Ж
30
41
2
3
4
5
6
7
8
9
Ў10
Џ11
А12
Б13
В14
Г15
Д16
Е17"
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
љBі
'__inference_model_2_layer_call_fn_48262input_3"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
'__inference_model_2_layer_call_fn_49655inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
'__inference_model_2_layer_call_fn_49764inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
'__inference_model_2_layer_call_fn_49185input_3"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
B__inference_model_2_layer_call_and_return_conditional_losses_49969inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
B__inference_model_2_layer_call_and_return_conditional_losses_50195inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
B__inference_model_2_layer_call_and_return_conditional_losses_49308input_3"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
B__inference_model_2_layer_call_and_return_conditional_losses_49431input_3"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
ЪBЧ
#__inference_signature_wrapper_49546input_3"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
+__inference_rescaling_2_layer_call_fn_50200inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
F__inference_rescaling_2_layer_call_and_return_conditional_losses_50208inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
нBк
)__inference_conv2d_18_layer_call_fn_50217inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
D__inference_conv2d_18_layer_call_and_return_conditional_losses_50227inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ћBј
6__inference_batch_normalization_34_layer_call_fn_50240inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
6__inference_batch_normalization_34_layer_call_fn_50253inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_50271inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_50289inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рBн
,__inference_activation_2_layer_call_fn_50294inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
G__inference_activation_2_layer_call_and_return_conditional_losses_50299inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBр
/__inference_max_pooling2d_6_layer_call_fn_50304inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_50309inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
юBы
)__inference_dropout_6_layer_call_fn_50314inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
)__inference_dropout_6_layer_call_fn_50319inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_dropout_6_layer_call_and_return_conditional_losses_50324inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_dropout_6_layer_call_and_return_conditional_losses_50336inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
Q
N0
O1
P2
Q3
R4
S5
T6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ђBя
+__inference_Bottleneck1_layer_call_fn_50385X"К
БВ­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
ђBя
+__inference_Bottleneck1_layer_call_fn_50434X"К
БВ­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
B
F__inference_Bottleneck1_layer_call_and_return_conditional_losses_50518X"К
БВ­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
B
F__inference_Bottleneck1_layer_call_and_return_conditional_losses_50602X"К
БВ­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
џ	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
 non_trainable_variables
Ёlayers
Ђmetrics
 Ѓlayer_regularization_losses
Єlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Х
Ѕtrace_0
Іtrace_12
(__inference_expbatch_layer_call_fn_50986
(__inference_expbatch_layer_call_fn_50999Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЅtrace_0zІtrace_1
ћ
Їtrace_0
Јtrace_12Р
C__inference_expbatch_layer_call_and_return_conditional_losses_51017
C__inference_expbatch_layer_call_and_return_conditional_losses_51035Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЇtrace_0zЈtrace_1
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Љnon_trainable_variables
Њlayers
Ћmetrics
 Ќlayer_regularization_losses
­layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Щ
Гtrace_0
Дtrace_12
*__inference_depthbatch_layer_call_fn_51048
*__inference_depthbatch_layer_call_fn_51061Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zГtrace_0zДtrace_1
џ
Еtrace_0
Жtrace_12Ф
E__inference_depthbatch_layer_call_and_return_conditional_losses_51079
E__inference_depthbatch_layer_call_and_return_conditional_losses_51097Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЕtrace_0zЖtrace_1
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
Ђ	variables
Ѓtrainable_variables
Єregularization_losses
І__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
Ч
Сtrace_0
Тtrace_12
)__inference_projbatch_layer_call_fn_51110
)__inference_projbatch_layer_call_fn_51123Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zСtrace_0zТtrace_1
§
Уtrace_0
Фtrace_12Т
D__inference_projbatch_layer_call_and_return_conditional_losses_51141
D__inference_projbatch_layer_call_and_return_conditional_losses_51159Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zУtrace_0zФtrace_1
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
Љ	variables
Њtrainable_variables
Ћregularization_losses
­__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
Щ
Ъtrace_0
Ыtrace_12
*__inference_shortbatch_layer_call_fn_51172
*__inference_shortbatch_layer_call_fn_51185Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЪtrace_0zЫtrace_1
џ
Ьtrace_0
Эtrace_12Ф
E__inference_shortbatch_layer_call_and_return_conditional_losses_51203
E__inference_shortbatch_layer_call_and_return_conditional_losses_51221Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЬtrace_0zЭtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBр
/__inference_max_pooling2d_7_layer_call_fn_50607inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_50612inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
юBы
)__inference_dropout_7_layer_call_fn_50617inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
)__inference_dropout_7_layer_call_fn_50622inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_dropout_7_layer_call_and_return_conditional_losses_50627inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_dropout_7_layer_call_and_return_conditional_losses_50639inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
`
Ў0
Џ1
А2
Б3
В4
Г5
Д6
Е7"
trackable_list_wrapper
Q
h0
i1
j2
k3
l4
m5
n6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ђBя
+__inference_Bottleneck2_layer_call_fn_50688X"К
БВ­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
ђBя
+__inference_Bottleneck2_layer_call_fn_50737X"К
БВ­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
B
F__inference_Bottleneck2_layer_call_and_return_conditional_losses_50821X"К
БВ­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
B
F__inference_Bottleneck2_layer_call_and_return_conditional_losses_50905X"К
БВ­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
0
 0
Ё1"
trackable_list_wrapper
0
 0
Ё1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
@
Ђ0
Ѓ1
Ў2
Џ3"
trackable_list_wrapper
0
Ђ0
Ѓ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
Х
иtrace_0
йtrace_12
(__inference_expbatch_layer_call_fn_51234
(__inference_expbatch_layer_call_fn_51247Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zиtrace_0zйtrace_1
ћ
кtrace_0
лtrace_12Р
C__inference_expbatch_layer_call_and_return_conditional_losses_51265
C__inference_expbatch_layer_call_and_return_conditional_losses_51283Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zкtrace_0zлtrace_1
 "
trackable_list_wrapper
0
Є0
Ѕ1"
trackable_list_wrapper
0
Є0
Ѕ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
з	variables
иtrainable_variables
йregularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
@
І0
Ї1
А2
Б3"
trackable_list_wrapper
0
І0
Ї1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
о	variables
пtrainable_variables
рregularization_losses
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
Щ
цtrace_0
чtrace_12
*__inference_depthbatch_layer_call_fn_51296
*__inference_depthbatch_layer_call_fn_51309Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zцtrace_0zчtrace_1
џ
шtrace_0
щtrace_12Ф
E__inference_depthbatch_layer_call_and_return_conditional_losses_51327
E__inference_depthbatch_layer_call_and_return_conditional_losses_51345Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zшtrace_0zщtrace_1
 "
trackable_list_wrapper
0
Ј0
Љ1"
trackable_list_wrapper
0
Ј0
Љ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
х	variables
цtrainable_variables
чregularization_losses
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
@
Њ0
Ћ1
В2
Г3"
trackable_list_wrapper
0
Њ0
Ћ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
яnon_trainable_variables
№layers
ёmetrics
 ђlayer_regularization_losses
ѓlayer_metrics
ь	variables
эtrainable_variables
юregularization_losses
№__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
Ч
єtrace_0
ѕtrace_12
)__inference_projbatch_layer_call_fn_51358
)__inference_projbatch_layer_call_fn_51371Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zєtrace_0zѕtrace_1
§
іtrace_0
їtrace_12Т
D__inference_projbatch_layer_call_and_return_conditional_losses_51389
D__inference_projbatch_layer_call_and_return_conditional_losses_51407Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zіtrace_0zїtrace_1
 "
trackable_list_wrapper
@
Ќ0
­1
Д2
Е3"
trackable_list_wrapper
0
Ќ0
­1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
јnon_trainable_variables
љlayers
њmetrics
 ћlayer_regularization_losses
ќlayer_metrics
ѓ	variables
єtrainable_variables
ѕregularization_losses
ї__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
Щ
§trace_0
ўtrace_12
*__inference_shortbatch_layer_call_fn_51420
*__inference_shortbatch_layer_call_fn_51433Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z§trace_0zўtrace_1
џ
џtrace_0
trace_12Ф
E__inference_shortbatch_layer_call_and_return_conditional_losses_51451
E__inference_shortbatch_layer_call_and_return_conditional_losses_51469Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zџtrace_0ztrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBр
/__inference_max_pooling2d_8_layer_call_fn_50910inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_50915inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
юBы
)__inference_dropout_8_layer_call_fn_50920inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
)__inference_dropout_8_layer_call_fn_50925inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_dropout_8_layer_call_and_return_conditional_losses_50930inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_dropout_8_layer_call_and_return_conditional_losses_50942inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
нBк
)__inference_flatten_2_layer_call_fn_50947inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
D__inference_flatten_2_layer_call_and_return_conditional_losses_50953inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
'__inference_dense_2_layer_call_fn_50962inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_dense_2_layer_call_and_return_conditional_losses_50973inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBъ
(__inference_expbatch_layer_call_fn_50986inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
(__inference_expbatch_layer_call_fn_50999inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_expbatch_layer_call_and_return_conditional_losses_51017inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_expbatch_layer_call_and_return_conditional_losses_51035inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яBь
*__inference_depthbatch_layer_call_fn_51048inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
*__inference_depthbatch_layer_call_fn_51061inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_depthbatch_layer_call_and_return_conditional_losses_51079inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_depthbatch_layer_call_and_return_conditional_losses_51097inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
юBы
)__inference_projbatch_layer_call_fn_51110inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
)__inference_projbatch_layer_call_fn_51123inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_projbatch_layer_call_and_return_conditional_losses_51141inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_projbatch_layer_call_and_return_conditional_losses_51159inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яBь
*__inference_shortbatch_layer_call_fn_51172inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
*__inference_shortbatch_layer_call_fn_51185inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_shortbatch_layer_call_and_return_conditional_losses_51203inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_shortbatch_layer_call_and_return_conditional_losses_51221inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ў0
Џ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBъ
(__inference_expbatch_layer_call_fn_51234inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
(__inference_expbatch_layer_call_fn_51247inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_expbatch_layer_call_and_return_conditional_losses_51265inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_expbatch_layer_call_and_return_conditional_losses_51283inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
А0
Б1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яBь
*__inference_depthbatch_layer_call_fn_51296inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
*__inference_depthbatch_layer_call_fn_51309inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_depthbatch_layer_call_and_return_conditional_losses_51327inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_depthbatch_layer_call_and_return_conditional_losses_51345inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
В0
Г1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
юBы
)__inference_projbatch_layer_call_fn_51358inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
)__inference_projbatch_layer_call_fn_51371inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_projbatch_layer_call_and_return_conditional_losses_51389inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_projbatch_layer_call_and_return_conditional_losses_51407inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
Д0
Е1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яBь
*__inference_shortbatch_layer_call_fn_51420inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
*__inference_shortbatch_layer_call_fn_51433inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_shortbatch_layer_call_and_return_conditional_losses_51451inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_shortbatch_layer_call_and_return_conditional_losses_51469inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
/:-2Adam/conv2d_18/kernel/m
!:2Adam/conv2d_18/bias/m
/:-2#Adam/batch_normalization_34/gamma/m
.:,2"Adam/batch_normalization_34/beta/m
&:$		2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
9:72!Adam/Bottleneck1/expconv/kernel/m
+:)2Adam/Bottleneck1/expconv/bias/m
-:+2!Adam/Bottleneck1/expbatch/gamma/m
,:*2 Adam/Bottleneck1/expbatch/beta/m
E:C2-Adam/Bottleneck1/depthconv/depthwise_kernel/m
-:+2!Adam/Bottleneck1/depthconv/bias/m
/:-2#Adam/Bottleneck1/depthbatch/gamma/m
.:,2"Adam/Bottleneck1/depthbatch/beta/m
::82"Adam/Bottleneck1/projconv/kernel/m
,:*2 Adam/Bottleneck1/projconv/bias/m
.:,2"Adam/Bottleneck1/projbatch/gamma/m
-:+2!Adam/Bottleneck1/projbatch/beta/m
/:-2#Adam/Bottleneck1/shortbatch/gamma/m
.:,2"Adam/Bottleneck1/shortbatch/beta/m
9:702!Adam/Bottleneck2/expconv/kernel/m
+:)02Adam/Bottleneck2/expconv/bias/m
-:+02!Adam/Bottleneck2/expbatch/gamma/m
,:*02 Adam/Bottleneck2/expbatch/beta/m
E:C02-Adam/Bottleneck2/depthconv/depthwise_kernel/m
-:+02!Adam/Bottleneck2/depthconv/bias/m
/:-02#Adam/Bottleneck2/depthbatch/gamma/m
.:,02"Adam/Bottleneck2/depthbatch/beta/m
::802"Adam/Bottleneck2/projconv/kernel/m
,:*2 Adam/Bottleneck2/projconv/bias/m
.:,2"Adam/Bottleneck2/projbatch/gamma/m
-:+2!Adam/Bottleneck2/projbatch/beta/m
/:-2#Adam/Bottleneck2/shortbatch/gamma/m
.:,2"Adam/Bottleneck2/shortbatch/beta/m
/:-2Adam/conv2d_18/kernel/v
!:2Adam/conv2d_18/bias/v
/:-2#Adam/batch_normalization_34/gamma/v
.:,2"Adam/batch_normalization_34/beta/v
&:$		2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
9:72!Adam/Bottleneck1/expconv/kernel/v
+:)2Adam/Bottleneck1/expconv/bias/v
-:+2!Adam/Bottleneck1/expbatch/gamma/v
,:*2 Adam/Bottleneck1/expbatch/beta/v
E:C2-Adam/Bottleneck1/depthconv/depthwise_kernel/v
-:+2!Adam/Bottleneck1/depthconv/bias/v
/:-2#Adam/Bottleneck1/depthbatch/gamma/v
.:,2"Adam/Bottleneck1/depthbatch/beta/v
::82"Adam/Bottleneck1/projconv/kernel/v
,:*2 Adam/Bottleneck1/projconv/bias/v
.:,2"Adam/Bottleneck1/projbatch/gamma/v
-:+2!Adam/Bottleneck1/projbatch/beta/v
/:-2#Adam/Bottleneck1/shortbatch/gamma/v
.:,2"Adam/Bottleneck1/shortbatch/beta/v
9:702!Adam/Bottleneck2/expconv/kernel/v
+:)02Adam/Bottleneck2/expconv/bias/v
-:+02!Adam/Bottleneck2/expbatch/gamma/v
,:*02 Adam/Bottleneck2/expbatch/beta/v
E:C02-Adam/Bottleneck2/depthconv/depthwise_kernel/v
-:+02!Adam/Bottleneck2/depthconv/bias/v
/:-02#Adam/Bottleneck2/depthbatch/gamma/v
.:,02"Adam/Bottleneck2/depthbatch/beta/v
::802"Adam/Bottleneck2/projconv/kernel/v
,:*2 Adam/Bottleneck2/projconv/bias/v
.:,2"Adam/Bottleneck2/projbatch/gamma/v
-:+2!Adam/Bottleneck2/projbatch/beta/v
/:-2#Adam/Bottleneck2/shortbatch/gamma/v
.:,2"Adam/Bottleneck2/shortbatch/beta/vь
F__inference_Bottleneck1_layer_call_and_return_conditional_losses_50518Ё,BЂ?
(Ђ%
# 
Xџџџџџџџџџ88
Њ

trainingp "-Ђ*
# 
0џџџџџџџџџ88
 ь
F__inference_Bottleneck1_layer_call_and_return_conditional_losses_50602Ё,BЂ?
(Ђ%
# 
Xџџџџџџџџџ88
Њ

trainingp"-Ђ*
# 
0џџџџџџџџџ88
 Ф
+__inference_Bottleneck1_layer_call_fn_50385,BЂ?
(Ђ%
# 
Xџџџџџџџџџ88
Њ

trainingp " џџџџџџџџџ88Ф
+__inference_Bottleneck1_layer_call_fn_50434,BЂ?
(Ђ%
# 
Xџџџџџџџџџ88
Њ

trainingp" џџџџџџџџџ88ь
F__inference_Bottleneck2_layer_call_and_return_conditional_losses_50821Ё, ЁЂЃЎЏЄЅІЇАБЈЉЊЋВГЌ­ДЕBЂ?
(Ђ%
# 
Xџџџџџџџџџ
Њ

trainingp "-Ђ*
# 
0џџџџџџџџџ
 ь
F__inference_Bottleneck2_layer_call_and_return_conditional_losses_50905Ё, ЁЂЃЎЏЄЅІЇАБЈЉЊЋВГЌ­ДЕBЂ?
(Ђ%
# 
Xџџџџџџџџџ
Њ

trainingp"-Ђ*
# 
0џџџџџџџџџ
 Ф
+__inference_Bottleneck2_layer_call_fn_50688, ЁЂЃЎЏЄЅІЇАБЈЉЊЋВГЌ­ДЕBЂ?
(Ђ%
# 
Xџџџџџџџџџ
Њ

trainingp " џџџџџџџџџФ
+__inference_Bottleneck2_layer_call_fn_50737, ЁЂЃЎЏЄЅІЇАБЈЉЊЋВГЌ­ДЕBЂ?
(Ђ%
# 
Xџџџџџџџџџ
Њ

trainingp" џџџџџџџџџј
 __inference__wrapped_model_47183гb'(1234 ЁЂЃЎЏЄЅІЇАБЈЉЊЋВГЌ­ДЕ:Ђ7
0Ђ-
+(
input_3џџџџџџџџџрр
Њ "1Њ.
,
dense_2!
dense_2џџџџџџџџџГ
G__inference_activation_2_layer_call_and_return_conditional_losses_50299h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџpp
Њ "-Ђ*
# 
0џџџџџџџџџpp
 
,__inference_activation_2_layer_call_fn_50294[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџpp
Њ " џџџџџџџџџppь
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_502711234MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ь
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_502891234MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ф
6__inference_batch_normalization_34_layer_call_fn_502401234MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџФ
6__inference_batch_normalization_34_layer_call_fn_502531234MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЖ
D__inference_conv2d_18_layer_call_and_return_conditional_losses_50227n'(9Ђ6
/Ђ,
*'
inputsџџџџџџџџџрр
Њ "-Ђ*
# 
0џџџџџџџџџpp
 
)__inference_conv2d_18_layer_call_fn_50217a'(9Ђ6
/Ђ,
*'
inputsџџџџџџџџџрр
Њ " џџџџџџџџџppЅ
B__inference_dense_2_layer_call_and_return_conditional_losses_50973_0Ђ-
&Ђ#
!
inputsџџџџџџџџџ	
Њ "%Ђ"

0џџџџџџџџџ
 }
'__inference_dense_2_layer_call_fn_50962R0Ђ-
&Ђ#
!
inputsџџџџџџџџџ	
Њ "џџџџџџџџџф
E__inference_depthbatch_layer_call_and_return_conditional_losses_51079MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ф
E__inference_depthbatch_layer_call_and_return_conditional_losses_51097MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ф
E__inference_depthbatch_layer_call_and_return_conditional_losses_51327ІЇАБMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 ф
E__inference_depthbatch_layer_call_and_return_conditional_losses_51345ІЇАБMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 М
*__inference_depthbatch_layer_call_fn_51048MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџМ
*__inference_depthbatch_layer_call_fn_51061MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџМ
*__inference_depthbatch_layer_call_fn_51296ІЇАБMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0М
*__inference_depthbatch_layer_call_fn_51309ІЇАБMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0Д
D__inference_dropout_6_layer_call_and_return_conditional_losses_50324l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ88
p 
Њ "-Ђ*
# 
0џџџџџџџџџ88
 Д
D__inference_dropout_6_layer_call_and_return_conditional_losses_50336l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ88
p
Њ "-Ђ*
# 
0џџџџџџџџџ88
 
)__inference_dropout_6_layer_call_fn_50314_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ88
p 
Њ " џџџџџџџџџ88
)__inference_dropout_6_layer_call_fn_50319_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ88
p
Њ " џџџџџџџџџ88Д
D__inference_dropout_7_layer_call_and_return_conditional_losses_50627l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ "-Ђ*
# 
0џџџџџџџџџ
 Д
D__inference_dropout_7_layer_call_and_return_conditional_losses_50639l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ "-Ђ*
# 
0џџџџџџџџџ
 
)__inference_dropout_7_layer_call_fn_50617_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ " џџџџџџџџџ
)__inference_dropout_7_layer_call_fn_50622_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ " џџџџџџџџџД
D__inference_dropout_8_layer_call_and_return_conditional_losses_50930l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ "-Ђ*
# 
0џџџџџџџџџ
 Д
D__inference_dropout_8_layer_call_and_return_conditional_losses_50942l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ "-Ђ*
# 
0џџџџџџџџџ
 
)__inference_dropout_8_layer_call_fn_50920_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ " џџџџџџџџџ
)__inference_dropout_8_layer_call_fn_50925_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ " џџџџџџџџџт
C__inference_expbatch_layer_call_and_return_conditional_losses_51017MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 т
C__inference_expbatch_layer_call_and_return_conditional_losses_51035MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 т
C__inference_expbatch_layer_call_and_return_conditional_losses_51265ЂЃЎЏMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 т
C__inference_expbatch_layer_call_and_return_conditional_losses_51283ЂЃЎЏMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
 К
(__inference_expbatch_layer_call_fn_50986MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџК
(__inference_expbatch_layer_call_fn_50999MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџК
(__inference_expbatch_layer_call_fn_51234ЂЃЎЏMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0К
(__inference_expbatch_layer_call_fn_51247ЂЃЎЏMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ0Љ
D__inference_flatten_2_layer_call_and_return_conditional_losses_50953a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ	
 
)__inference_flatten_2_layer_call_fn_50947T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "џџџџџџџџџ	э
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_50309RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Х
/__inference_max_pooling2d_6_layer_call_fn_50304RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџэ
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_50612RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Х
/__inference_max_pooling2d_7_layer_call_fn_50607RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџэ
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_50915RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Х
/__inference_max_pooling2d_8_layer_call_fn_50910RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
B__inference_model_2_layer_call_and_return_conditional_losses_49308Яb'(1234 ЁЂЃЎЏЄЅІЇАБЈЉЊЋВГЌ­ДЕBЂ?
8Ђ5
+(
input_3џџџџџџџџџрр
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
B__inference_model_2_layer_call_and_return_conditional_losses_49431Яb'(1234 ЁЂЃЎЏЄЅІЇАБЈЉЊЋВГЌ­ДЕBЂ?
8Ђ5
+(
input_3џџџџџџџџџрр
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
B__inference_model_2_layer_call_and_return_conditional_losses_49969Юb'(1234 ЁЂЃЎЏЄЅІЇАБЈЉЊЋВГЌ­ДЕAЂ>
7Ђ4
*'
inputsџџџџџџџџџрр
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
B__inference_model_2_layer_call_and_return_conditional_losses_50195Юb'(1234 ЁЂЃЎЏЄЅІЇАБЈЉЊЋВГЌ­ДЕAЂ>
7Ђ4
*'
inputsџџџџџџџџџрр
p

 
Њ "%Ђ"

0џџџџџџџџџ
 ю
'__inference_model_2_layer_call_fn_48262Тb'(1234 ЁЂЃЎЏЄЅІЇАБЈЉЊЋВГЌ­ДЕBЂ?
8Ђ5
+(
input_3џџџџџџџџџрр
p 

 
Њ "џџџџџџџџџю
'__inference_model_2_layer_call_fn_49185Тb'(1234 ЁЂЃЎЏЄЅІЇАБЈЉЊЋВГЌ­ДЕBЂ?
8Ђ5
+(
input_3џџџџџџџџџрр
p

 
Њ "џџџџџџџџџэ
'__inference_model_2_layer_call_fn_49655Сb'(1234 ЁЂЃЎЏЄЅІЇАБЈЉЊЋВГЌ­ДЕAЂ>
7Ђ4
*'
inputsџџџџџџџџџрр
p 

 
Њ "џџџџџџџџџэ
'__inference_model_2_layer_call_fn_49764Сb'(1234 ЁЂЃЎЏЄЅІЇАБЈЉЊЋВГЌ­ДЕAЂ>
7Ђ4
*'
inputsџџџџџџџџџрр
p

 
Њ "џџџџџџџџџу
D__inference_projbatch_layer_call_and_return_conditional_losses_51141MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 у
D__inference_projbatch_layer_call_and_return_conditional_losses_51159MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 у
D__inference_projbatch_layer_call_and_return_conditional_losses_51389ЊЋВГMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 у
D__inference_projbatch_layer_call_and_return_conditional_losses_51407ЊЋВГMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Л
)__inference_projbatch_layer_call_fn_51110MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЛ
)__inference_projbatch_layer_call_fn_51123MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЛ
)__inference_projbatch_layer_call_fn_51358ЊЋВГMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЛ
)__inference_projbatch_layer_call_fn_51371ЊЋВГMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЖ
F__inference_rescaling_2_layer_call_and_return_conditional_losses_50208l9Ђ6
/Ђ,
*'
inputsџџџџџџџџџрр
Њ "/Ђ,
%"
0џџџџџџџџџрр
 
+__inference_rescaling_2_layer_call_fn_50200_9Ђ6
/Ђ,
*'
inputsџџџџџџџџџрр
Њ ""џџџџџџџџџррф
E__inference_shortbatch_layer_call_and_return_conditional_losses_51203MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ф
E__inference_shortbatch_layer_call_and_return_conditional_losses_51221MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ф
E__inference_shortbatch_layer_call_and_return_conditional_losses_51451Ќ­ДЕMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ф
E__inference_shortbatch_layer_call_and_return_conditional_losses_51469Ќ­ДЕMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 М
*__inference_shortbatch_layer_call_fn_51172MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџМ
*__inference_shortbatch_layer_call_fn_51185MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџМ
*__inference_shortbatch_layer_call_fn_51420Ќ­ДЕMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџМ
*__inference_shortbatch_layer_call_fn_51433Ќ­ДЕMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
#__inference_signature_wrapper_49546оb'(1234 ЁЂЃЎЏЄЅІЇАБЈЉЊЋВГЌ­ДЕEЂB
Ђ 
;Њ8
6
input_3+(
input_3џџџџџџџџџрр"1Њ.
,
dense_2!
dense_2џџџџџџџџџ