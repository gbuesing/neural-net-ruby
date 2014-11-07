require_relative 'neural_net'
require 'zlib'

# This neural net performs OCR on handwritten digits from the MNIST dataset
# MNIST datafiles can be downloaded here: http://yann.lecun.com/exdb/mnist/

mnist_images_file = 'mnist/train-images-idx3-ubyte.gz'
mnist_labels_file = 'mnist/train-labels-idx1-ubyte.gz'

unless File.exist?(mnist_images_file) && File.exist?(mnist_labels_file)
  raise "Missing MNIST datafiles\nDownload MNIST datafiles from: \nhttp://yann.lecun.com/exdb/mnist/"
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

data.shuffle!

train_size = (ARGV[0] || 100).to_i
test_size = 100

# maps input to float between 0 and 1
normalize = -> (val, fromLow, fromHigh, toLow, toHigh) {  (val - fromLow) * (toHigh - toLow) / (fromHigh - fromLow).to_f }

dataset = data.slice(0,train_size + test_size).map do |row|
  image = row[0].unpack('C*')
  image = image.map {|v| normalize.(v, 0, 256, 0, 1)}
  [image, row[1]]
end

train_data = dataset.slice(0, train_size)
test_data = dataset.slice(train_size, test_size)


nn = NeuralNet.new [28*28,100,10]

puts "Testing the untrained network..."

success, failure = 0,0
test_data.each do |input, expected|
  output = nn.run input
  predicted = (0..9).max_by{|i| output[i]}
  actual = (0..9).max_by{|i| expected[i]}
  predicted == actual ? success += 1 : failure += 1
end

puts "Untrained prediction success: #{success}, failure: #{failure}"


puts "\nTraining the network with #{train_size} data samples...\n\n"
t = Time.now
result = nn.train(train_data, log_every: 1, iterations: 100)

puts "\nDone training the network: #{result[:iterations]} iterations, error #{result[:error].round(5)}, #{(Time.now - t).round(1)}s"


puts "\nTesting the trained network..."

success, failure = 0,0
test_data.each do |input, expected|
  output = nn.run input
  predicted = (0..9).max_by{|i| output[i]}
  actual = (0..9).max_by{|i| expected[i]}
  predicted == actual ? success += 1 : failure += 1
end

puts "Trained prediction success: #{success}, failure: #{failure}"
