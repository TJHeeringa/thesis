З
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
6
Pow
x"T
y"T
z"T"
Ttype:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8˫
k
grelu/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_name	grelu/w
d
grelu/w/Read/ReadVariableOpReadVariableOpgrelu/w*
_output_shapes
:	?*
dtype0
g
grelu/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	grelu/b
`
grelu/b/Read/ReadVariableOpReadVariableOpgrelu/b*
_output_shapes	
:?*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	?*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
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
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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
y
Adam/grelu/w/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_nameAdam/grelu/w/m
r
"Adam/grelu/w/m/Read/ReadVariableOpReadVariableOpAdam/grelu/w/m*
_output_shapes
:	?*
dtype0
u
Adam/grelu/b/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameAdam/grelu/b/m
n
"Adam/grelu/b/m/Read/ReadVariableOpReadVariableOpAdam/grelu/b/m*
_output_shapes	
:?*
dtype0
?
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/output/kernel/m
~
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes
:	?*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:*
dtype0
y
Adam/grelu/w/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_nameAdam/grelu/w/v
r
"Adam/grelu/w/v/Read/ReadVariableOpReadVariableOpAdam/grelu/w/v*
_output_shapes
:	?*
dtype0
u
Adam/grelu/b/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameAdam/grelu/b/v
n
"Adam/grelu/b/v/Read/ReadVariableOpReadVariableOpAdam/grelu/b/v*
_output_shapes	
:?*
dtype0
?
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/output/kernel/v
~
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes
:	?*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
`
	w

b
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?
iter

beta_1

beta_2
	decay
learning_rate	m.
m/m0m1	v2
v3v4v5
 

	0

1
2
3

	0

1
2
3
?
layer_regularization_losses
metrics
layer_metrics
regularization_losses
trainable_variables

layers
non_trainable_variables
	variables
 
NL
VARIABLE_VALUEgrelu/w1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEgrelu/b1layer_with_weights-0/b/.ATTRIBUTES/VARIABLE_VALUE
 

	0

1

	0

1
?
layer_regularization_losses
 metrics
!layer_metrics
regularization_losses
trainable_variables

"layers
#non_trainable_variables
	variables
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
$layer_regularization_losses
%metrics
&layer_metrics
regularization_losses
trainable_variables

'layers
(non_trainable_variables
	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

)0
 

0
1
 
 
 
 
 
 
 
 
 
 
 
4
	*total
	+count
,	variables
-	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

,	variables
qo
VARIABLE_VALUEAdam/grelu/w/mMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/grelu/b/mMlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/grelu/w/vMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/grelu/b/vMlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_grelu_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_grelu_inputgrelu/wgrelu/boutput/kerneloutput/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_2869081
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamegrelu/w/Read/ReadVariableOpgrelu/b/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"Adam/grelu/w/m/Read/ReadVariableOp"Adam/grelu/b/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp"Adam/grelu/w/v/Read/ReadVariableOp"Adam/grelu/b/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst* 
Tin
2	*
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
GPU 2J 8? *)
f$R"
 __inference__traced_save_2869330
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegrelu/wgrelu/boutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/grelu/w/mAdam/grelu/b/mAdam/output/kernel/mAdam/output/bias/mAdam/grelu/w/vAdam/grelu/b/vAdam/output/kernel/vAdam/output/bias/v*
Tin
2*
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
GPU 2J 8? *,
f'R%
#__inference__traced_restore_2869397??
?
?
K__inference_sequential_244_layer_call_and_return_conditional_losses_2869100
grelu_input(
$grelu_matmul_readvariableop_resource%
!grelu_add_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??grelu/MatMul/ReadVariableOp?grelu/add/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
grelu/MatMul/ReadVariableOpReadVariableOp$grelu_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
grelu/MatMul/ReadVariableOp?
grelu/MatMulMatMulgrelu_input#grelu/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
grelu/MatMul?
grelu/add/ReadVariableOpReadVariableOp!grelu_add_readvariableop_resource*
_output_shapes	
:?*
dtype02
grelu/add/ReadVariableOp?
	grelu/addAddV2grelu/MatMul:product:0 grelu/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	grelu/addb

grelu/ReluRelugrelu/add:z:0*
T0*(
_output_shapes
:??????????2

grelu/Relu_
grelu/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
grelu/Pow/y?
	grelu/PowPowgrelu/Relu:activations:0grelu/Pow/y:output:0*
T0*(
_output_shapes
:??????????2
	grelu/Pow?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMulgrelu/Pow:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAdd?
IdentityIdentityoutput/BiasAdd:output:0^grelu/MatMul/ReadVariableOp^grelu/add/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2:
grelu/MatMul/ReadVariableOpgrelu/MatMul/ReadVariableOp24
grelu/add/ReadVariableOpgrelu/add/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:T P
'
_output_shapes
:?????????
%
_user_specified_namegrelu_input
?
?
0__inference_sequential_244_layer_call_fn_2869132
grelu_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgrelu_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_sequential_244_layer_call_and_return_conditional_losses_28690202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_namegrelu_input
?
?
"__inference__wrapped_model_2868929
grelu_input7
3sequential_244_grelu_matmul_readvariableop_resource4
0sequential_244_grelu_add_readvariableop_resource8
4sequential_244_output_matmul_readvariableop_resource9
5sequential_244_output_biasadd_readvariableop_resource
identity??*sequential_244/grelu/MatMul/ReadVariableOp?'sequential_244/grelu/add/ReadVariableOp?,sequential_244/output/BiasAdd/ReadVariableOp?+sequential_244/output/MatMul/ReadVariableOp?
*sequential_244/grelu/MatMul/ReadVariableOpReadVariableOp3sequential_244_grelu_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*sequential_244/grelu/MatMul/ReadVariableOp?
sequential_244/grelu/MatMulMatMulgrelu_input2sequential_244/grelu/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_244/grelu/MatMul?
'sequential_244/grelu/add/ReadVariableOpReadVariableOp0sequential_244_grelu_add_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential_244/grelu/add/ReadVariableOp?
sequential_244/grelu/addAddV2%sequential_244/grelu/MatMul:product:0/sequential_244/grelu/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_244/grelu/add?
sequential_244/grelu/ReluRelusequential_244/grelu/add:z:0*
T0*(
_output_shapes
:??????????2
sequential_244/grelu/Relu}
sequential_244/grelu/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
sequential_244/grelu/Pow/y?
sequential_244/grelu/PowPow'sequential_244/grelu/Relu:activations:0#sequential_244/grelu/Pow/y:output:0*
T0*(
_output_shapes
:??????????2
sequential_244/grelu/Pow?
+sequential_244/output/MatMul/ReadVariableOpReadVariableOp4sequential_244_output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02-
+sequential_244/output/MatMul/ReadVariableOp?
sequential_244/output/MatMulMatMulsequential_244/grelu/Pow:z:03sequential_244/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_244/output/MatMul?
,sequential_244/output/BiasAdd/ReadVariableOpReadVariableOp5sequential_244_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_244/output/BiasAdd/ReadVariableOp?
sequential_244/output/BiasAddBiasAdd&sequential_244/output/MatMul:product:04sequential_244/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_244/output/BiasAdd?
IdentityIdentity&sequential_244/output/BiasAdd:output:0+^sequential_244/grelu/MatMul/ReadVariableOp(^sequential_244/grelu/add/ReadVariableOp-^sequential_244/output/BiasAdd/ReadVariableOp,^sequential_244/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2X
*sequential_244/grelu/MatMul/ReadVariableOp*sequential_244/grelu/MatMul/ReadVariableOp2R
'sequential_244/grelu/add/ReadVariableOp'sequential_244/grelu/add/ReadVariableOp2\
,sequential_244/output/BiasAdd/ReadVariableOp,sequential_244/output/BiasAdd/ReadVariableOp2Z
+sequential_244/output/MatMul/ReadVariableOp+sequential_244/output/MatMul/ReadVariableOp:T P
'
_output_shapes
:?????????
%
_user_specified_namegrelu_input
?
?
0__inference_sequential_244_layer_call_fn_2869145
grelu_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgrelu_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_sequential_244_layer_call_and_return_conditional_losses_28690472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_namegrelu_input
?
|
'__inference_grelu_layer_call_fn_2869231

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_grelu_layer_call_and_return_conditional_losses_28689462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_sequential_244_layer_call_and_return_conditional_losses_2869119
grelu_input(
$grelu_matmul_readvariableop_resource%
!grelu_add_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??grelu/MatMul/ReadVariableOp?grelu/add/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
grelu/MatMul/ReadVariableOpReadVariableOp$grelu_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
grelu/MatMul/ReadVariableOp?
grelu/MatMulMatMulgrelu_input#grelu/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
grelu/MatMul?
grelu/add/ReadVariableOpReadVariableOp!grelu_add_readvariableop_resource*
_output_shapes	
:?*
dtype02
grelu/add/ReadVariableOp?
	grelu/addAddV2grelu/MatMul:product:0 grelu/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	grelu/addb

grelu/ReluRelugrelu/add:z:0*
T0*(
_output_shapes
:??????????2

grelu/Relu_
grelu/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
grelu/Pow/y?
	grelu/PowPowgrelu/Relu:activations:0grelu/Pow/y:output:0*
T0*(
_output_shapes
:??????????2
	grelu/Pow?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMulgrelu/Pow:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAdd?
IdentityIdentityoutput/BiasAdd:output:0^grelu/MatMul/ReadVariableOp^grelu/add/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2:
grelu/MatMul/ReadVariableOpgrelu/MatMul/ReadVariableOp24
grelu/add/ReadVariableOpgrelu/add/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:T P
'
_output_shapes
:?????????
%
_user_specified_namegrelu_input
?
?
K__inference_sequential_244_layer_call_and_return_conditional_losses_2869183

inputs(
$grelu_matmul_readvariableop_resource%
!grelu_add_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??grelu/MatMul/ReadVariableOp?grelu/add/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
grelu/MatMul/ReadVariableOpReadVariableOp$grelu_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
grelu/MatMul/ReadVariableOp?
grelu/MatMulMatMulinputs#grelu/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
grelu/MatMul?
grelu/add/ReadVariableOpReadVariableOp!grelu_add_readvariableop_resource*
_output_shapes	
:?*
dtype02
grelu/add/ReadVariableOp?
	grelu/addAddV2grelu/MatMul:product:0 grelu/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	grelu/addb

grelu/ReluRelugrelu/add:z:0*
T0*(
_output_shapes
:??????????2

grelu/Relu_
grelu/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
grelu/Pow/y?
	grelu/PowPowgrelu/Relu:activations:0grelu/Pow/y:output:0*
T0*(
_output_shapes
:??????????2
	grelu/Pow?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMulgrelu/Pow:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAdd?
IdentityIdentityoutput/BiasAdd:output:0^grelu/MatMul/ReadVariableOp^grelu/add/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2:
grelu/MatMul/ReadVariableOpgrelu/MatMul/ReadVariableOp24
grelu/add/ReadVariableOpgrelu/add/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_2869081
grelu_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgrelu_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_28689292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_namegrelu_input
?

?
B__inference_grelu_layer_call_and_return_conditional_losses_2869222

inputs"
matmul_readvariableop_resource
add_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
add/ReadVariableOpt
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
addP
ReluReluadd:z:0*
T0*(
_output_shapes
:??????????2
ReluS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
Pow/yh
PowPowRelu:activations:0Pow/y:output:0*
T0*(
_output_shapes
:??????????2
Pow?
IdentityIdentityPow:z:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
}
(__inference_output_layer_call_fn_2869250

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_28689722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_output_layer_call_and_return_conditional_losses_2869241

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
0__inference_sequential_244_layer_call_fn_2869209

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_sequential_244_layer_call_and_return_conditional_losses_28690472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?Q
?	
#__inference__traced_restore_2869397
file_prefix
assignvariableop_grelu_w
assignvariableop_1_grelu_b$
 assignvariableop_2_output_kernel"
assignvariableop_3_output_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count&
"assignvariableop_11_adam_grelu_w_m&
"assignvariableop_12_adam_grelu_b_m,
(assignvariableop_13_adam_output_kernel_m*
&assignvariableop_14_adam_output_bias_m&
"assignvariableop_15_adam_grelu_w_v&
"assignvariableop_16_adam_grelu_b_v,
(assignvariableop_17_adam_output_kernel_v*
&assignvariableop_18_adam_output_bias_v
identity_20??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-0/b/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_grelu_wIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_grelu_bIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_output_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_output_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_adam_grelu_w_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_adam_grelu_b_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp(assignvariableop_13_adam_output_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_output_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_adam_grelu_w_vIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_adam_grelu_b_vIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_output_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_output_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_189
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19?
Identity_20IdentityIdentity_19:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_20"#
identity_20Identity_20:output:0*a
_input_shapesP
N: :::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
K__inference_sequential_244_layer_call_and_return_conditional_losses_2869020

inputs
grelu_2869009
grelu_2869011
output_2869014
output_2869016
identity??grelu/StatefulPartitionedCall?output/StatefulPartitionedCall?
grelu/StatefulPartitionedCallStatefulPartitionedCallinputsgrelu_2869009grelu_2869011*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_grelu_layer_call_and_return_conditional_losses_28689462
grelu/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall&grelu/StatefulPartitionedCall:output:0output_2869014output_2869016*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_28689722 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^grelu/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2>
grelu/StatefulPartitionedCallgrelu/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_sequential_244_layer_call_and_return_conditional_losses_2869047

inputs
grelu_2869036
grelu_2869038
output_2869041
output_2869043
identity??grelu/StatefulPartitionedCall?output/StatefulPartitionedCall?
grelu/StatefulPartitionedCallStatefulPartitionedCallinputsgrelu_2869036grelu_2869038*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_grelu_layer_call_and_return_conditional_losses_28689462
grelu/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall&grelu/StatefulPartitionedCall:output:0output_2869041output_2869043*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_28689722 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0^grelu/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2>
grelu/StatefulPartitionedCallgrelu/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_sequential_244_layer_call_fn_2869196

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_sequential_244_layer_call_and_return_conditional_losses_28690202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_sequential_244_layer_call_and_return_conditional_losses_2869164

inputs(
$grelu_matmul_readvariableop_resource%
!grelu_add_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??grelu/MatMul/ReadVariableOp?grelu/add/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
grelu/MatMul/ReadVariableOpReadVariableOp$grelu_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
grelu/MatMul/ReadVariableOp?
grelu/MatMulMatMulinputs#grelu/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
grelu/MatMul?
grelu/add/ReadVariableOpReadVariableOp!grelu_add_readvariableop_resource*
_output_shapes	
:?*
dtype02
grelu/add/ReadVariableOp?
	grelu/addAddV2grelu/MatMul:product:0 grelu/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	grelu/addb

grelu/ReluRelugrelu/add:z:0*
T0*(
_output_shapes
:??????????2

grelu/Relu_
grelu/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
grelu/Pow/y?
	grelu/PowPowgrelu/Relu:activations:0grelu/Pow/y:output:0*
T0*(
_output_shapes
:??????????2
	grelu/Pow?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMulgrelu/Pow:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAdd?
IdentityIdentityoutput/BiasAdd:output:0^grelu/MatMul/ReadVariableOp^grelu/add/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2:
grelu/MatMul/ReadVariableOpgrelu/MatMul/ReadVariableOp24
grelu/add/ReadVariableOpgrelu/add/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?/
?
 __inference__traced_save_2869330
file_prefix&
"savev2_grelu_w_read_readvariableop&
"savev2_grelu_b_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_adam_grelu_w_m_read_readvariableop-
)savev2_adam_grelu_b_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop-
)savev2_adam_grelu_w_v_read_readvariableop-
)savev2_adam_grelu_b_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-0/b/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0"savev2_grelu_w_read_readvariableop"savev2_grelu_b_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_adam_grelu_w_m_read_readvariableop)savev2_adam_grelu_b_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop)savev2_adam_grelu_w_v_read_readvariableop)savev2_adam_grelu_b_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *"
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes}
{: :	?:?:	?:: : : : : : : :	?:?:	?::	?:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
?

?
B__inference_grelu_layer_call_and_return_conditional_losses_2868946

inputs"
matmul_readvariableop_resource
add_readvariableop_resource
identity??MatMul/ReadVariableOp?add/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
add/ReadVariableOpt
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
addP
ReluReluadd:z:0*
T0*(
_output_shapes
:??????????2
ReluS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
Pow/yh
PowPowRelu:activations:0Pow/y:output:0*
T0*(
_output_shapes
:??????????2
Pow?
IdentityIdentityPow:z:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
C__inference_output_layer_call_and_return_conditional_losses_2868972

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
grelu_input4
serving_default_grelu_input:0?????????:
output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?Z
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
6__call__
*7&call_and_return_all_conditional_losses
8_default_save_signature"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_244", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_244", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "grelu_input"}}, {"class_name": "GReLU", "config": {"name": "grelu", "trainable": true, "dtype": "float32", "units": 500, "power": 5}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_244", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "grelu_input"}}, {"class_name": "GReLU", "config": {"name": "grelu", "trainable": true, "dtype": "float32", "units": 500, "power": 5}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	w

b
regularization_losses
trainable_variables
	variables
	keras_api
9__call__
*:&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GReLU", "name": "grelu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "grelu", "trainable": true, "dtype": "float32", "units": 500, "power": 5}, "build_input_shape": {"class_name": "TensorShape", "items": [25, 1]}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
;__call__
*<&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}}, "build_input_shape": {"class_name": "TensorShape", "items": [25, 500]}}
?
iter

beta_1

beta_2
	decay
learning_rate	m.
m/m0m1	v2
v3v4v5"
	optimizer
 "
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
?
layer_regularization_losses
metrics
layer_metrics
regularization_losses
trainable_variables

layers
non_trainable_variables
	variables
6__call__
8_default_save_signature
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
,
=serving_default"
signature_map
:	?2grelu/w
:?2grelu/b
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
?
layer_regularization_losses
 metrics
!layer_metrics
regularization_losses
trainable_variables

"layers
#non_trainable_variables
	variables
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
 :	?2output/kernel
:2output/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
$layer_regularization_losses
%metrics
&layer_metrics
regularization_losses
trainable_variables

'layers
(non_trainable_variables
	variables
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	*total
	+count
,	variables
-	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
*0
+1"
trackable_list_wrapper
-
,	variables"
_generic_user_object
:	?2Adam/grelu/w/m
:?2Adam/grelu/b/m
%:#	?2Adam/output/kernel/m
:2Adam/output/bias/m
:	?2Adam/grelu/w/v
:?2Adam/grelu/b/v
%:#	?2Adam/output/kernel/v
:2Adam/output/bias/v
?2?
0__inference_sequential_244_layer_call_fn_2869132
0__inference_sequential_244_layer_call_fn_2869209
0__inference_sequential_244_layer_call_fn_2869145
0__inference_sequential_244_layer_call_fn_2869196?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_sequential_244_layer_call_and_return_conditional_losses_2869164
K__inference_sequential_244_layer_call_and_return_conditional_losses_2869119
K__inference_sequential_244_layer_call_and_return_conditional_losses_2869100
K__inference_sequential_244_layer_call_and_return_conditional_losses_2869183?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference__wrapped_model_2868929?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"
grelu_input?????????
?2?
'__inference_grelu_layer_call_fn_2869231?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_grelu_layer_call_and_return_conditional_losses_2869222?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_output_layer_call_fn_2869250?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_output_layer_call_and_return_conditional_losses_2869241?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_2869081grelu_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_2868929m	
4?1
*?'
%?"
grelu_input?????????
? "/?,
*
output ?
output??????????
B__inference_grelu_layer_call_and_return_conditional_losses_2869222]	
/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? {
'__inference_grelu_layer_call_fn_2869231P	
/?,
%?"
 ?
inputs?????????
? "????????????
C__inference_output_layer_call_and_return_conditional_losses_2869241]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_output_layer_call_fn_2869250P0?-
&?#
!?
inputs??????????
? "???????????
K__inference_sequential_244_layer_call_and_return_conditional_losses_2869100k	
<?9
2?/
%?"
grelu_input?????????
p

 
? "%?"
?
0?????????
? ?
K__inference_sequential_244_layer_call_and_return_conditional_losses_2869119k	
<?9
2?/
%?"
grelu_input?????????
p 

 
? "%?"
?
0?????????
? ?
K__inference_sequential_244_layer_call_and_return_conditional_losses_2869164f	
7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
K__inference_sequential_244_layer_call_and_return_conditional_losses_2869183f	
7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
0__inference_sequential_244_layer_call_fn_2869132^	
<?9
2?/
%?"
grelu_input?????????
p

 
? "???????????
0__inference_sequential_244_layer_call_fn_2869145^	
<?9
2?/
%?"
grelu_input?????????
p 

 
? "???????????
0__inference_sequential_244_layer_call_fn_2869196Y	
7?4
-?*
 ?
inputs?????????
p

 
? "???????????
0__inference_sequential_244_layer_call_fn_2869209Y	
7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
%__inference_signature_wrapper_2869081|	
C?@
? 
9?6
4
grelu_input%?"
grelu_input?????????"/?,
*
output ?
output?????????