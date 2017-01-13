// basic neural-net in C++

// this should serve as a template
// and for teaching purposes
//
// more documentation needed

// To Run:
// vim ../trainingData.txt
// g++ basicNN.cpp -o basicNN
// ./basicNN > out.txt
// vim out.txt

#include <vector>
#include <iostream>
#include <cstdlib>

using namespace std;


struct Connection
{
  double weight;
  double deltaWeight;
};

class Neuron; // ref

typedef vector<Neuron> Layer;


// **************** class Neuron ****************

class Neuron
{
public:
  Neuron(unsigned numOutputs, unsigned m_myIndex);
  void setOutputVal(double val) { m_outputVal = val; }
  double getOutputVal(void) const { return m_outputVal; }
  void feedForward(const Layer &prevLayer);
  void calcOutputGradients(double targetVal);
  void calcHiddenGradients(const Layer &nextLayer);
  void updateInputWeights(Layer &prevLayer);

private:
  static double eta;    // [0.0..1.0] overall net training rate
  static double alpha;  // [0.0..n] multiplier of last weight change (momentum)
  static double transferFunction(double x);
  static double transferFunctionDerivative(double x);
  static double randomWeight(void) { return rand() / double(RAND_MAX); }
  double sumDOW(const Layer &nextLayer) const;
  double m_outputVal;
  vector<Connection> m_outputWeights;
  unsigned m_myIndex;
  double m_gradient;
};

// Note:
// Different types of problems do better with different etas and alphas.
// Beware of overfitting your data --context is key.
// These values are good to experiment with.

double Neuron::eta = 0.15; // learning rate
double Neuron::alpha = 0.5; // momentum

void Neuron::updateInputWeights(Layer &prevLayer)
{
  // The weights to be updated are in the Connection container
  // in the neurons in the preceding layer.

  for (unsigned n = 0; n < prevLayer.size(); ++n) {
    Neuron &neuron = prevLayer[n];
    double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

    double newDeltaWeight =
          // Individual input, magnified by gradient and train rate:
          eta
          * neuron.getOutputVal()
          * m_gradient
          // Also add momentum = a fraction of the previous delta weight:
          + alpha
          * oldDeltaWeight;

    neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
    neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
  }
}

double Neuron::sumDOW(const Layer &nextLayer)
{
  double sum = 0.0;

  // Sum our contributions of the errors of the nodes we feed.

  for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
    sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
  }

  return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
  double dow = sumDOW(nextLayer);

  \\ This is one of several different ways to define gradient.
  \\ It may be a good idea to try alternatives --depending on context.

  m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
  double delta - targetVal - m_outputVal;

  \\ This is one of several different ways to define gradient.
  \\ It may be a good idea to try alternatives --depending on context.

  m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
  // This needs to be something like a step function.
  // However, since we're using a smoothed-version (I need to take derivatives)
  // we need something like a sigmoid curve.

  // Here we are using hyperbolic tangent.
  // tanh - output range: [-1.0..1.0]
  return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
  // Quick Approximation: d/dx tanh(x) ~ 1 - x^2
  return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
  double sum = 0.0;

  // Sum the previous layer's outputs (this layer's inputs)
  // Include the bias node from previous layer

  for (unsigned n = 0; n < prevLayer.size(); ++n) {
    sum += prevLayer[n].getOutputVal() *
            prevLayer[n].m_outputWeights[m_myIndex].weight;
  }

  m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs)
{
  for (unsigned c = 0; c < numOutputs; ++c) {
    m_outputWeights.push_back(Connection());
    m_outputWeights.back().weight = randomWeight();
  }
}

// **************** class Net ****************

class Net
{
public:
  Net(vector<unsigned> topology);
  void feedForward(const vector<double> &inputVals);
  void backProp(const std::vector<double> &targetVals);
  void getResults(vector<double> &resultVals) const;

private:
  vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
  double m_error;
  double m_recentAverageError;
  double m_recentAverageSmoothingFactor;
};

void Net::getResults(vector<double> &resultVals) const
{
  resultVals.clear();

  for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
    resultVals.push_back(m_layers.back()[n].getOutputVal());
  }
}

void Net::backProp(const vector<double> &targetVals)
{
  // Calculate overall net error
  // E.g. RMSE of ouput neuron errors
  // This is what we try to mimimize (where the "learning" happens)
  // (in Kaggle competitions you can change this to fit the scoring evaluation)

  Layer &outputLayer = m_layers.back();
  m_error = 0.0;

  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
    double delta = targetVals[n] - outputLayer[n].getOutputVal();
    m_error += delta * delta;
  }
  m_error /= outputLayer.size() - 1; // get average error squared
  m_error = sqrt(m_error); // RMSE

  // Implement a recent average measurement:

  m_recentAverageError =
      (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
      / (m_recentAverageSmoothingFactor + 1.0);

  // Calculate output layer gradients

  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
    outputLayer[n].calcOutputGradients(targetVals[n]);
  }

  // Calculate gradients on hidden Layers

  for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
    Layer &hiddenLayer = m_layers[layerNum];
    Layer &nextLayer = m_layers[layerNum + 1];

    for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
      hiddenLayer[n].calcHiddenGradients(nextLayer);
    }
  }

  // For all layers from outputs to first hidden layer.
  // update connection weights

  for (unsigned layerNum = m_layers.size() - 1;  layerNum > 0; --layerNum) {
    Layer &layer = m_layers[layerNum];
    Layer &prevLayer = m_layers[layerNum - 1];

    for (unsigned n = 0; n < layer.size() - 1;  ++n) {
      layer[n].updateInputWeights(prevLayer);
    }
  }
}

void Net::feedForward(const vector<double> &inputVals)
{
  assert(inputVals.size() == m_layers[0].size() - 1); // minus one for bias neuron

  // Assign {latch} the input values into the input neurons
  for (unsigned i = 0; i < inputVals.size(); ++i) {
    m_layers[0][i].setOutputVal(inputVals[i]);
  }

  // Forward propogate
  for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
    Layer &prevLayer = m_layers[layerNum - 1];
    for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
      m_layers[layerNum][n].feedForward(prevLayer);
    }
  }
}

Net::Net(const vector<unsigned> &topology)
{
  unsigned numLayers = topology.size();
  for (unsigned layerNum=0; layerNum < numLayers; ++layerNum) {
    m_layers.push_back(Layer());
    unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

    // made a new layer --now we fill it with neurons
    // add bias neurons to layer ( hence <= instead of < )
    for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
      m_layers.back().pushback(Neuron(numOutputs, neuronNum));
      cout << "Made a Neuron!" << endl;
    }

    // Force the bias node's output value to 1.0. It's the last neuron in each layer.
    m_layers.back().back().setOutputVal(1.0);
  }
}

int main()
{
  // e.g., { 3, 2, 1 }
  vector<unsigned> topology;
  topology.push_back(3);  // 3+1 Neurons in 0th Layer
  topology.push_back(2);  // 2+1 Neurons in 1st Layer
  topology.push_back(1);  // 1+1 Neurons in 2nd Layer
  Net myNet(topology);    // Total: 9 Neurons / 3 Layers

  vector<double> inputVals;
  myNet.feedForward(inputVals);

  vector<double> targetVals;
  myNet.backProp(targetVals);

  vector<double> resultVals;
  myNet.getResults(resultVals);
};
