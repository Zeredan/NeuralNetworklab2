package org.example.domain

import org.example.core.derSigmoid
import org.example.core.mseLoss
import org.example.core.sigmoid

class MyNeuralNetwork(
    private val h1: Neurone = Neurone.default(2),
    private val h2: Neurone = Neurone.default(2),
    private val o1: Neurone = Neurone.default(2)
) {
    class Neurone(
        val weights: MutableList<Float>,
        var b: Float,
        val func: (value: Float) -> Float = ::sigmoid
    ) {
        data class NeuroneResult(
            val outputSignal: Float,
            val sum: Float
        )

        operator fun invoke(signals: List<Float>): NeuroneResult {
            check(signals.size == weights.size)
            return (signals.zip(weights).sumOf { it.first * it.second.toDouble() } + b).toFloat().run {
                //println("a: $this")
                NeuroneResult(
                    outputSignal = func(this),
                    sum = this
                )
            }
        }

        companion object {
            fun default(weights: Int): Neurone = Neurone(
                MutableList(weights) { Math.random().toFloat() },
                Math.random().toFloat()
            )
        }
    }

    private var w1: Float
        get() = h1.weights[0]
        set(value) {h1.weights[0] = value}

    private var w2: Float
        get() = h1.weights[1]
        set(value) {h1.weights[1] = value}

    private var w3: Float
        get() = h2.weights[0]
        set(value) {h2.weights[0] = value}

    private var w4: Float
        get() = h2.weights[1]
        set(value) {h2.weights[1] = value}

    private var w5: Float
        get() = o1.weights[0]
        set(value) {o1.weights[0] = value}

    private var w6: Float
        get() = o1.weights[1]
        set(value) {o1.weights[1] = value}

    private var b1: Float
        get() = h1.b
        set(value) {h1.b = value}

    private var b2: Float
        get() = h2.b
        set(value) {h2.b = value}

    private var b3: Float
        get() = o1.b
        set(value) {o1.b = value}

    operator fun invoke(signals: List<Float>) : Neurone.NeuroneResult{
        val h1 = h1(signals).outputSignal
        val h2 = h2(signals).outputSignal
        val o1 = o1(listOf(h1, h2))
        //println("$h1 | $h2 | $signals")
        return o1
    }

    fun teach(
        data: List<Pair<List<Float>, Float>>,
        epochs: Int = 1000,
        learningRate: Float = 0.1f,
        derFunc: (Float) -> Float = ::derSigmoid
    ) {
        println("$w1 | $w2 | $w3 | $w4 | $w5 | $w6")
        repeat(epochs) { epoch ->
            data.forEach { (signals, yTrue) ->
                val (h1, h1Sum) = h1(signals)
                val (h2, h2Sum) = h2(signals)

                val (o1, o1Sum) = o1(listOf(h1, h2))


                val L_o1 = -2 * (yTrue - o1)

                /// o1 neurone
                val o1_w5 = h1 * derFunc(o1Sum)
                val o1_w6 = h2 * derFunc(o1Sum)
                val o1_b3 = derFunc(o1Sum)

                val o1_h1 = w5 * derFunc(o1Sum)
                val o1_h2 = w6 * derFunc(o1Sum)

                /// h1 neurone
                val h1_w1 = signals[0] * derFunc(h1Sum)
                val h1_w2 = signals[1] * derFunc(h1Sum)
                val h1_b1 = derFunc(h1Sum)

                /// h2 neurone
                val h2_w3 = signals[0] * derFunc(h2Sum)
                val h2_w4 = signals[1] * derFunc(h2Sum)
                val h2_b2 = derFunc(h2Sum)

                //println("$h1Sum $h2Sum $o1Sum $o1 ___$L_o1 | $o1_w5 | $o1_w6 | $o1_b3 | $o1_h1 | $o1_h2 ||| $h1_w1 | $h1_w2 | $h1_b1 ||| $h2_w3 | $h2_w4 | $h2_b2")
                //throw Exception()
                /// updating weights
                w1 -= learningRate * L_o1 * o1_h1 * h1_w1
                w2 -= learningRate * L_o1 * o1_h1 * h1_w2
                b1 -= learningRate * L_o1 * o1_h1 * h1_b1

                w3 -= learningRate * L_o1 * o1_h2 * h2_w3
                w4 -= learningRate * L_o1 * o1_h2 * h2_w4
                b2 -= learningRate * L_o1 * o1_h2 * h2_b2

                w5 -= learningRate * L_o1 * o1_w5
                w6 -= learningRate * L_o1 * o1_w6
                b3 -= learningRate * L_o1 * o1_b3
            }
            /// after epoch learning step checking what error is
            val results = data.map { this(it.first).outputSignal }

            val mseLoss = results mseLoss data.map { it.second }

            if ((epoch + 1) % 10 == 0) println("Epoch: ${epoch + 1}; MSE_LOSS: $mseLoss | $w1 | $w2 | $w3 | $w4 | $w5 | $w6")
        }
    }
}