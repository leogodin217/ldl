from ldl.predictions import predict_digit
import sure
import pytest
import numpy as np


def test_predict_digit_predicts_zero():
    output = np.array([.75, .6, .5, .744, .3, 0, 0, 0, 0, .7])
    prediction = predict_digit(output)
    prediction.should.equal(0)


def test_predict_digit_predicts_one():
    output = np.array([.75, .8, .5, .744, .3, 0, 0, 0, 0, .7])
    prediction = predict_digit(output)
    prediction.should.equal(1)


def test_predict_digit_predicts_two():
    output = np.array([.75, .6, .9, .744, .3, 0, 0, 0, 0, .7])
    prediction = predict_digit(output)
    prediction.should.equal(2)


def test_predict_digit_predicts_three():
    output = np.array([.75, .6, .5, .8, .3, 0, 0, 0, 0, .7])
    prediction = predict_digit(output)
    prediction.should.equal(3)


def test_predict_digit_predicts_four():
    output = np.array([.75, .6, .5, .744, .9, 0, 0, 0, 0, .7])
    prediction = predict_digit(output)
    prediction.should.equal(4)


def test_predict_digit_predicts_five():
    output = np.array([.75, .6, .5, .744, .3, .9, 0, 0, 0, .7])
    prediction = predict_digit(output)
    prediction.should.equal(5)


def test_predict_digit_predicts_six():
    output = np.array([.75, .6, .5, .744, .3, 0, .9, 0, 0, .7])
    prediction = predict_digit(output)
    prediction.should.equal(6)


def test_predict_digit_predicts_seven():
    output = np.array([.75, .6, .5, .744, .3, 0, 0, .9, 0, .7])
    prediction = predict_digit(output)
    prediction.should.equal(7)


def test_predict_digit_predicts_eight():
    output = np.array([.75, .6, .5, .744, .3, 0, 0, 0, .9, .7])
    prediction = predict_digit(output)
    prediction.should.equal(8)


def test_predict_digit_predicts_nine():
    output = np.array([.75, .6, .5, .744, .3, 0, 0, 0, .9, .99])
    prediction = predict_digit(output)
    prediction.should.equal(9)



