"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { CheckCircle, XCircle, AlertCircle } from "lucide-react"

export function AlgorithmComparison() {
  const algorithms = [
    {
      name: "Deep Q-Network (DQN)",
      type: "Value-Based",
      description: "Uses deep neural networks to approximate Q-values for state-action pairs.",
      pros: [
        "Sample efficient for discrete actions",
        "Stable learning with experience replay",
        "Well-established and widely used",
        "Good for environments with clear value functions",
      ],
      cons: [
        "Limited to discrete action spaces",
        "Can overestimate Q-values",
        "Requires careful hyperparameter tuning",
        "May struggle with continuous control",
      ],
      bestFor: ["Discrete action environments", "Atari games", "Grid worlds", "Navigation tasks"],
      complexity: "Medium",
      sampleEfficiency: "High",
      stability: "High",
    },
    {
      name: "Proximal Policy Optimization (PPO)",
      type: "Policy-Based",
      description: "Uses clipped surrogate objective to ensure stable policy updates.",
      pros: [
        "Works with continuous and discrete actions",
        "More stable than vanilla policy gradients",
        "Simpler to implement than TRPO",
        "Good general-purpose algorithm",
      ],
      cons: [
        "Can be sample inefficient",
        "Sensitive to hyperparameters",
        "May get stuck in local optima",
        "Requires more computational resources",
      ],
      bestFor: [
        "Continuous control tasks",
        "Robotics applications",
        "Multi-agent environments",
        "Complex action spaces",
      ],
      complexity: "Medium",
      sampleEfficiency: "Medium",
      stability: "High",
    },
  ]

  const getIcon = (level: string) => {
    switch (level) {
      case "High":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "Medium":
        return <AlertCircle className="h-4 w-4 text-yellow-500" />
      case "Low":
        return <XCircle className="h-4 w-4 text-red-500" />
      default:
        return <AlertCircle className="h-4 w-4 text-gray-500" />
    }
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h3 className="text-2xl font-bold mb-2">Algorithm Comparison</h3>
        <p className="text-gray-600">Compare the characteristics and performance of DQN and PPO algorithms</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {algorithms.map((algo, index) => (
          <Card key={index} className="h-full">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-xl">{algo.name}</CardTitle>
                <Badge variant="outline">{algo.type}</Badge>
              </div>
              <CardDescription>{algo.description}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Performance Metrics */}
              <div>
                <h4 className="font-semibold mb-3">Performance Characteristics</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Complexity:</span>
                    <div className="flex items-center space-x-2">
                      {getIcon(algo.complexity)}
                      <span className="text-sm">{algo.complexity}</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Sample Efficiency:</span>
                    <div className="flex items-center space-x-2">
                      {getIcon(algo.sampleEfficiency)}
                      <span className="text-sm">{algo.sampleEfficiency}</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Stability:</span>
                    <div className="flex items-center space-x-2">
                      {getIcon(algo.stability)}
                      <span className="text-sm">{algo.stability}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Pros */}
              <div>
                <h4 className="font-semibold mb-2 text-green-700">Advantages</h4>
                <ul className="space-y-1">
                  {algo.pros.map((pro, i) => (
                    <li key={i} className="text-sm flex items-start">
                      <CheckCircle className="h-3 w-3 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                      {pro}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Cons */}
              <div>
                <h4 className="font-semibold mb-2 text-red-700">Limitations</h4>
                <ul className="space-y-1">
                  {algo.cons.map((con, i) => (
                    <li key={i} className="text-sm flex items-start">
                      <XCircle className="h-3 w-3 text-red-500 mr-2 mt-0.5 flex-shrink-0" />
                      {con}
                    </li>
                  ))}
                </ul>
              </div>

              {/* ε-greedy policy section for DQN only */}
              {algo.name === "Deep Q-Network (DQN)" && (
                <div>
                  <h4 className="font-semibold mb-2">*ε-greedy policy</h4>
                  <ul className="space-y-1">
                    <li className="text-sm flex items-start">
                      <span className="mr-2">•</span>ε probability to act randomly
                    </li>
                    <li className="text-sm flex items-start">
                      <span className="mr-2">•</span>
                      1-ε probability to act greedily (to choose action with highest Q-value)
                    </li>
                  </ul>
                  <div className="mt-3">
                    <p className="text-sm font-bold">*innovated ε random action specific for this environment</p>
                    <ul className="space-y-1 mt-2">
                      <li className="text-sm flex items-start">
                        <strong>1)</strong>
                        <span className="ml-2">
                          we avoid using the same action (to wall and package) if the agent is crashing into solid as
                          before
                        </span>
                      </li>
                      <li className="text-sm flex items-start">
                        <strong>2)</strong>
                        <span className="ml-2">
                          meanwhile, if the agent is crashing and the agent is carrying package, we try the turn actions
                          if it is not trying before
                        </span>
                      </li>
                      <li className="text-sm flex items-start">
                        <strong>3)</strong>
                        <span className="ml-2">
                          if the agent continueously crash into solid for 5 times, as the reward becomes penalty (in
                          environment module), we also freeze the action for 10 future steps
                        </span>
                      </li>
                      <li className="text-sm flex items-start">
                        <strong>4)</strong>
                        <span className="ml-2">
                          <u>
                            Not Random Action: if the agent is not bouncing to hard things, we just continue the action
                          </u>
                        </span>
                      </li>
                    </ul>
                  </div>
                </div>
              )}

              {/* Best For */}
              <div>
                <h4 className="font-semibold mb-2">Best Suited For</h4>
                <div className="flex flex-wrap gap-2">
                  {algo.bestFor.map((use, i) => (
                    <Badge key={i} variant="secondary" className="text-xs">
                      {use}
                    </Badge>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Comparison Table */}
      <Card>
        <CardHeader>
          <CardTitle>Side-by-Side Comparison</CardTitle>
          <CardDescription>Quick comparison of key features and characteristics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2">Feature</th>
                  <th className="text-center py-2">DQN</th>
                  <th className="text-center py-2">PPO</th>
                </tr>
              </thead>
              <tbody className="space-y-2">
                <tr className="border-b">
                  <td className="py-2 font-medium">Type</td>
                  <td className="text-center py-2">Value-based (Critic)</td>
                  <td className="text-center py-2">Policy-based (Actor-Critic)</td>
                </tr>
                <tr className="border-b">
                  <td className="py-2 font-medium">Action Space</td>
                  <td className="text-center py-2">Discrete Only</td>
                  <td className="text-center py-2">Continuous & Discrete</td>
                </tr>
                <tr className="border-b">
                  <td className="py-2 font-medium">Learning Type</td>
                  <td className="text-center py-2">Off-Policy</td>
                  <td className="text-center py-2">On-Policy</td>
                </tr>
                <tr className="border-b">
                  <td className="py-2 font-medium">Memory Usage</td>
                  <td className="text-center py-2">High (Replay Buffer)</td>
                  <td className="text-center py-2">Low</td>
                </tr>
                <tr className="border-b">
                  <td className="py-2 font-medium">Training Speed</td>
                  <td className="text-center py-2">Fast</td>
                  <td className="text-center py-2">Moderate</td>
                </tr>
                <tr>
                  <td className="py-2 font-medium">Implementation</td>
                  <td className="text-center py-2">Moderate</td>
                  <td className="text-center py-2">Moderate</td>
                </tr>
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
