package org.example.core

import kotlin.math.exp
import kotlin.math.sqrt

internal fun sigmoid(value: Float) : Float {
    return (1f / (1 + exp(-value)))
}

internal fun derSigmoid(value: Float) : Float {
    return sigmoid(value).run {
        this * (1 - this)
    }
}

internal infix fun List<Float>.mseLoss(lst: List<Float>) : Float {
    return sqrt(this.zip(lst).sumOf { (it.second - it.first).run{this * this}.toDouble() }).toFloat()
}