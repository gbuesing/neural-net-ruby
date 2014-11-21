Neural Net (in Ruby)
===

A [feedforward neural network](http://en.wikipedia.org/wiki/Feedforward_neural_network) with [resilient backpropagation](http://en.wikipedia.org/wiki/Rprop) (Rprop). It's ~250 loc, 100% Ruby, with no external dependencies.

This implementation trains significantly faster than [ai4r](https://github.com/SergioFierens/ai4r)'s backpropagation neural network, mainly because the Rprop training algorithm implemented here is much faster than the non-batch backpropagation algorithm used in ai4r.

However, this implementation is significantly slower than [ruby-fann](https://github.com/tangledpath/ruby-fann), which wraps the FANN library, written in C. If you're looking for something production-ready, use ruby-fann.


Examples
---
- ```iris.rb```: solves a simple classification problem: predict the species of iris flower based on sepal and petal size.
- ```mnist.rb```: performs OCR on handwritten digits. Requires download of MNIST dataset; see instructions at top of file.


Sources and inspirations
---

- [Introduction to the Math of Neural Networks](http://www.amazon.com/Introduction-Math-Neural-Networks-Heaton-ebook/dp/B00845UQL6)
- [Thoughtful Machine Learning: A Test-Driven Approach](http://www.amazon.com/Thoughtful-Machine-Learning-Test-Driven-Approach/dp/1449374069)
- [Hacker's guide to Neural Networks](http://karpathy.github.io/neuralnets/)
- [https://github.com/harthur/brain](https://github.com/harthur/brain)
- [The RPROP Algorithm](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.21.1417&rep=rep1&type=pdf)
