#!/usr/bin/env ruby

require_relative '../neural_net'
require 'zlib'

# This neural net performs OCR on handwritten digits from the MNIST dataset
# MNIST datafiles can be downloaded here: http://yann.lecun.com/exdb/mnist/

mnist_images_file = 'examples/mnist/train-images-idx3-ubyte.gz'
mnist_labels_file = 'examples/mnist/train-labels-idx1-ubyte.gz'

unless File.exist?(mnist_images_file) && File.exist?(mnist_labels_file)
  raise "Missing MNIST datafiles\nMNIST datafiles must be present in an mnist/ directory\nDownload from: http://yann.lecun.com/exdb/mnist/"
end

# MNIST loading code adapted from here:
# https://github.com/shuyo/iir/blob/master/neural/mnist.rb
n_rows = n_cols = nil
images = []
labels = []
Zlib::GzipReader.open(mnist_images_file) do |f|
  magic, n_images = f.read(8).unpack('N2')
  raise 'This is not MNIST image file' if magic != 2051
  n_rows, n_cols = f.read(8).unpack('N2')
  n_images.times do
    images << f.read(n_rows * n_cols)
  end
end

Zlib::GzipReader.open(mnist_labels_file) do |f|
  magic, n_labels = f.read(8).unpack('N2')
  raise 'This is not MNIST label file' if magic != 2049
  labels = f.read(n_labels).unpack('C*')
end

# collate image and label data
data = images.map.with_index do |image, i|
  target = [0]*10
  target[labels[i]] = 1
  [image, target]
end

# data.shuffle!

train_size = (ARGV[0] || 100).to_i
test_size = 100
hidden_layer_size = (ARGV[1] || 25).to_i

# maps input to float between 0 and 1
normalize = -> (val, fromLow, fromHigh, toLow, toHigh) {  (val - fromLow) * (toHigh - toLow) / (fromHigh - fromLow).to_f }

x_data, y_data = [], []

data.slice(0,train_size + test_size).each do |row|
  image = row[0].unpack('C*')
  image = image.map {|v| normalize.(v, 0, 256, 0, 1)}
  x_data << image
  y_data << row[1]
end

x_train = x_data.slice(0, train_size)
y_train = y_data.slice(0, train_size)

x_test = x_data.slice(train_size, test_size)
y_test = y_data.slice(train_size, test_size)


puts "Initializing network with #{hidden_layer_size} hidden neurons."
nn = NeuralNet.new [28*28,hidden_layer_size, 50, 10]

error_rate = -> (errors, total) { ((errors / total.to_f) * 100).round }

mse = -> (actual, ideal) {
  errors = actual.zip(ideal).map {|a, i| a - i }
  (errors.inject(0) {|sum, err| sum += err**2}) / errors.length.to_f
}

decode_output = -> (output) { (0..9).max_by {|i| output[i]} }
prediction_success = -> (actual, ideal) { decode_output.(actual) == decode_output.(ideal) }

run_test = -> (nn, inputs, expected_outputs) {
  success, failure, errsum = 0,0,0
  inputs.each.with_index do |input, i|
    output = nn.run input
    prediction_success.(output, expected_outputs[i]) ? success += 1 : failure += 1
    errsum += mse.(output, expected_outputs[i])
  end
  [success, failure, errsum / inputs.length.to_f]
}

puts "Testing the untrained network..."

success, failure, avg_mse = run_test.(nn, x_test, y_test)

puts "Untrained classification success: #{success}, failure: #{failure} (classification error: #{error_rate.(failure, x_test.length)}%, mse: #{(avg_mse * 100).round(2)}%)"

puts "\nTraining the network with #{train_size} data samples...\n\n"
t = Time.now
result = nn.train(x_train, y_train, log_every: 1, max_iterations: 100, error_threshold:  0.01)

puts "\nDone training the network: #{result[:iterations]} iterations, #{(result[:error] * 100).round(2)}% mse, #{(Time.now - t).round(1)}s"

# # Marshal test
# dumpfile = 'mnist/network.dump'
# File.write(dumpfile, Marshal.dump(nn))
# nn = Marshal.load(File.read(dumpfile))

puts "\nTesting the trained network..."

success, failure, avg_mse = run_test.(nn, x_test, y_test)

puts "Trained classification success: #{success}, failure: #{failure} (classification error: #{error_rate.(failure, x_test.length)}%, mse: #{(avg_mse * 100).round(2)}%)"


# require_relative './image_grid'
# ImageGrid.new(nn.weights[1]).to_file 'examples/mnist/hidden_weights.png'
