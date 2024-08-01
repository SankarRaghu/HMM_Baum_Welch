using Random
using Statistics
using StatsBase
using Distributions
using LinearAlgebra
using Plots
plotlyjs()
using LaTeXStrings

function initProbabilities(nS::Int64, nO::Int64)
  # Initializing the initial probabilities of the states
  πs = (1 / nS) * ones(Float64, nS)

  # Initializing the transition probability matrix
  A = (1 / nS) * ones(Float64, nS, nS)

  # Initializing the emission probability matrix
  B = (1 / nO) * ones(Float64, nS, nO)

  return πs, A, B
end

function forwardPass(πs::Vector{Float64}, A::Matrix{Float64},
                     B::Matrix{Float64}, O::Vector{Int64},
                     nS::Int64, nTrain::Int64)
  c = zeros(Float64, nTrain)
  α = zeros(Float64, nS, nTrain)
  @inbounds for i in 1:nS
    α[i, 1] = πs[i] * B[i, O[1]]
  end
  c[1] = 1 / sum(α[:, 1])
  α[:, 1] *= c[1]

  @inbounds for t in 2:nTrain
    @inbounds for i in 1:nS
      sum_α = 0.0
      @inbounds for j in 1:nS
        sum_α += α[j, t - 1] * A[j, i]
      end
      α[i, t] = B[i, O[t]] * sum_α
    end
    c[t] = 1 / sum(α[:, t])
    α[:, t] *= c[t]
  end

  return α, c
end

function backwardPass(πs::Vector{Float64}, A::Matrix{Float64},
                      B::Matrix{Float64}, c::Vector{Float64}, O::Vector{Int64}, nS::Int64,
                      nTrain::Int64)
  β = ones(Float64, nS, nTrain)
  β[:, end] *= c[end]

  @inbounds for t in (nTrain - 1):-1:1
    @inbounds for i in 1:nS
      sum_β = 0.0
      @inbounds for j in 1:nS
        sum_β += β[j, t + 1] * A[i, j] * B[j, O[t + 1]]
      end
      β[i, t] = sum_β
    end
    β[:, t] *= c[t]
  end

  return β
end

function computeGamma(α::Matrix{Float64}, β::Matrix{Float64},
                      nS::Int64, nO::Int64, nTrain::Int64)
  γ = zeros(Float64, nS, nTrain)
  @inbounds for t in 1:nTrain
    @inbounds for i in 1:nS
      sum_αβ = 0.0
      @inbounds for j in 1:nS
        sum_αβ += α[j, t] * β[j, t]
      end
      γ[i, t] = α[i, t] * β[i, t] / sum_αβ
    end
  end

  return γ
end

function computeXi(α::Matrix{Float64}, β::Matrix{Float64}, A::Matrix{Float64},
                   B::Matrix{Float64}, O::Vector{Int64}, nS::Int64, nTrain::Int64)
  ξ = zeros(Float64, nS, nS, nTrain - 1)
  @inbounds for t in 1:(nTrain - 1)
    sum_ξ = 0.0
    @inbounds for k in 1:nS
      @inbounds for w in 1:nS
        sum_ξ += α[k, t] * A[k, w] * β[w, t + 1] * B[w, O[t + 1]]
      end
    end
    
    @inbounds for i = 1:nS
      @inbounds for j = 1:nS
       ξ[i, j, t] = α[i, t] * A[i, j] * β[j, t + 1] * B[j, O[t + 1]] / sum_ξ
      end
    end
  end

  return ξ
end

function baumWelchAlgorithm(O::Vector{Int64}, nS::Int64, nO::Int64,
                            nTrain::Int64)
  # Initializing the probabilities
  πs, A, B = initProbabilities(nS, nO)
  α, c = forwardPass(πs, A, B, O, nS, nTrain)
  β = backwardPass(πs, A, B, c, O, nS, nTrain)
  γ = computeGamma(α, β, nS, nO, nTrain)
  ξ = computeXi(α, β, A, B, O, nS, nTrain)
  display(ξ)
end

rng = MersenneTwister(1000)
w = pweights([0.1, 0.1, 0.3, 0.2, 0.3])

nS = 3
nO = 5
nTrain = 20
nTest = 10

observations = sample(rng, 1:nO, w, nTrain + nTest)

baumWelchAlgorithm(observations, nS, nO, nTrain)

default(; size=(1000, 1000))
scatter(1:(nTrain + nTest), observations; markercolor="red")
gui()

readline()