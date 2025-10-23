using COSMOAccelerators
using Clarabel
using Parameters
using LinearAlgebra
import LinearAlgebra:givensAlgorithm
using LinearAlgebra.LAPACK
import SparseArrays
using LinearMaps

const DefaultFloat = Float64
const DefaultInt = Int64

# Define an abstract type for an inverse linear operator
abstract type AbstractInvOp end