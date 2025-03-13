package org.example

import org.example.domain.MyNeuralNetwork

fun main() {
    val network = MyNeuralNetwork()
    //1 - Рост, 2 - Вес.
    //0 - Женщина/1 - Мужчина.
    network.teach(
        data = listOf(
            listOf(194f, 81f) to 1f,
            listOf(184f, 80f) to 1f,
            listOf(186f, 86f) to 1f,
            listOf(190f, 90f) to 1f,
            listOf(179f, 78f) to 1f,
            listOf(200f, 93f) to 1f,
            listOf(197f, 90f) to 1f,
            listOf(183f, 81f) to 1f,
            listOf(181f, 76f) to 1f,
            listOf(180f, 78f) to 1f,

            listOf(154f, 49f) to 0f,
            listOf(164f, 56f) to 0f,
            listOf(164f, 51f) to 0f,
            listOf(164f, 55f) to 0f,
            listOf(154f, 47f) to 0f,
            listOf(174f, 57f) to 0f,
            listOf(164f, 51f) to 0f,
            listOf(164f, 50f) to 0f,
            listOf(154f, 53f) to 0f,
            listOf(164f, 55f) to 0f,
        ).map {
            listOf(it.first[0] / 200f, it.first[1] / 90f) to it.second
        }
    )

    val maria = (listOf(170f / 200f, 51f / 90f) to 0f)
    val vladimir = listOf(194f, 81f) to 1f

    val mariaResult = network(maria.first).outputSignal
    val vladimirResult = network(vladimir.first).outputSignal

    println("mariaTrue: ${maria.second} | mariaPred: $mariaResult")
    println("vladimirTrue: ${vladimir.second} | vladimirPred: $vladimirResult")
}