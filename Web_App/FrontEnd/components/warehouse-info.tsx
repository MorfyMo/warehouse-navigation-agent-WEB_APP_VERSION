"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { AlertTriangle, MapPin, Package, Users } from "lucide-react"

interface WarehouseInfoProps {
  detailed?: boolean
}

export function WarehouseInfo({ detailed = false }: WarehouseInfoProps) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Package className="h-5 w-5 mr-2" />
            Warehouse Robot Navigation Environment
          </CardTitle>
          <CardDescription>
            Multi-agent reinforcement learning in a warehouse package management scenario
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Overview */}
          <div>
            <h4 className="font-semibold mb-3 flex items-center">
              <MapPin className="h-4 w-4 mr-2" />
              Environment Overview
            </h4>
            <p className="text-sm text-muted-foreground mb-4">
              In this warehouse environment, multiple robots work together to transport packages from storage shelves to
              the receiving region. The goal is to efficiently move all packages while following operational
              constraints.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Badge variant="outline" className="w-full justify-start">
                  <Package className="h-3 w-3 mr-2" />
                  Total Packages Exist: 146
                </Badge>
                <Badge variant="outline" className="w-full justify-start">
                  <Users className="h-3 w-3 mr-2" />
                  Multi-Agent System
                </Badge>
              </div>
              <div className="space-y-2">
                <Badge variant="outline" className="w-full justify-start">
                  One Package Per Robot
                </Badge>
                <Badge variant="outline" className="w-full justify-start">
                  Collaborative/Competitive Operations
                </Badge>
              </div>
            </div>
          </div>

          <Separator />

          {/* Environment Area Types */}
          <div>
            <h4 className="font-semibold mb-3 flex items-center">
              <Users className="h-4 w-4 mr-2" />
              Environment Component Types
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="p-4">
                <div className="flex items-center mb-2">
                  <div className="w-4 h-4 bg-black rounded-full mr-2"></div>
                  <h5 className="font-medium">Solid</h5>
                </div>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• This include the walls and shelves that store packages</li>
                  <li>• Prevent the agents from getting through the area</li>
                  <li>• Agents get penalized when crashing into the walls and shelves</li>
                </ul>
              </Card>

              <Card className="p-4">
                <div className="flex items-center mb-2">
                  <div className="w-4 h-4 bg-gray-400 rounded-full mr-2"></div>
                  <h5 className="font-medium">Empty</h5>
                </div>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• The clean ground without anything</li>
                  <li>• Allow the agents to move or move with packages</li>
                </ul>
              </Card>

              <Card className="p-4">
                <div className="flex items-center mb-2">
                  <div className="w-4 h-4 bg-yellow-600 rounded-full mr-2"></div>
                  <h5 className="font-medium">Packages</h5>
                </div>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Packages(large boxes) the agents required to move</li>
                  <li>• Picks up from the shelves</li>
                  <li>• Drops packages in transfer region</li>
                </ul>
              </Card>
            </div>
          </div>

          <div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 justify-start">
              <Card className="p-4">
                <div className="flex items-center mb-2">
                  <div className="w-4 h-4 bg-yellow-300 rounded-full mr-2"></div>
                  <h5 className="font-medium">Receiving Region</h5>
                </div>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Where normal agents and helper agents put packages on</li>
                  <li>• Package is delivered when put onto the receiving region</li>
                </ul>
              </Card>

              <Card className="p-4">
                <div className="flex items-center mb-2">
                  <div className="w-4 h-4 bg-blue-300 rounded-full mr-2"></div>
                  <h5 className="font-medium">Transfer Region</h5>
                </div>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• The "magic" region where the collector pass the package to</li>
                  <li>• The region from which the packages delivered are sent out for external delivery</li>
                  <li>• Reduce the burden of the receiving region</li>
                  <li>• Potentially prepare for future external delivery</li>
                </ul>
              </Card>


            </div>
          </div>

          <Separator />
          {/* Agent Types */}
          <div>
            <h4 className="font-semibold mb-3 flex items-center">
              <Users className="h-4 w-4 mr-2" />
              Agent Types
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="p-4">
                <div className="flex items-center mb-2">
                  <div className="w-4 h-4 bg-red-600 rounded-full mr-2"></div>
                  <h5 className="font-medium">Normal Agent</h5>
                </div>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Can pickup packages from shelves</li>
                  <li>• Can move packages in receiving region</li>
                  <li>• Can transfer packages to other agents</li>
                  <li>• Primary workforce for package handling</li>
                </ul>
              </Card>

              <Card className="p-4">
                <div className="flex items-center mb-2">
                  <div className="w-4 h-4 bg-blue-500 rounded-full mr-2"></div>
                  <h5 className="font-medium">Helper Agent</h5>
                </div>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Cannot pickup from shelves or receiving region directly</li>
                  <li>• Can only receive packages from other agents</li>
                  <li>• Assists in package transportation with normal agents</li>
                  {/* <li>• Helps optimize delivery routes</li> */}
                  <li>• Only cooperate with normal agents</li>
                </ul>
              </Card>

              <Card className="p-4">
                <div className="flex items-center mb-2">
                  <div className="w-4 h-4 bg-blue-800 rounded-full mr-2"></div>
                  <h5 className="font-medium">Collector Agent</h5>
                </div>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Limited to receiving & transfer regions</li>
                  <li>• Picks up from receiving region only</li>
                  <li>• Drops packages in transfer region</li>
                  <li>• No agent-to-agent transfers allowed</li>
                </ul>
              </Card>
            </div>
          </div>

          {detailed && (
            <>
              <Separator />

              {/* Rules and Constraints */}
              <div>
                <h4 className="font-semibold mb-3 flex items-center">
                  <AlertTriangle className="h-4 w-4 mr-2" />
                  Operational Rules
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h5 className="font-medium mb-2">Movement Constraints</h5>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      <li>• Robots can only carry one package at a time</li>
                      <li>• Packages can be repositioned within receiving region</li>
                      <li>• Transfer region is off-limits for normal/helper agents</li>
                      <li>• Collector agents cannot enter main warehouse area</li>
                    </ul>
                  </div>
                  <div>
                    <h5 className="font-medium mb-2">Delivery Requirements</h5>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      <li>• Packages must be dropped in the receiving region</li>
                      <li>• Collector reduces burden on receiving region</li>
                      <li>• Efficient coordination between agent types required</li>
                      <li>• Goal: Transport all 146 packages (you can adjust goal) successfully</li>
                    </ul>
                  </div>
                </div>
              </div>

              <Separator />

              {/* Reward Structure */}
              <div>
                <h4 className="font-semibold leading-tight">Main Reward Structure</h4>
                <p className="text-xs text-gray-400">for details, see Documention bar</p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h5 className="font-medium mb-2 mt-2 text-green-600">Positive Rewards</h5>
                    <ul className="text-sm text-gray-700 space-y-1">
                      <li>• +80✖️pickup_count: Package successfully delivered to transfer region</li>
                      <li>• +80✖️pickup_count: Package moved to receiving region</li>
                      <li>• +25✖️pickup_count(+85): Successful agent-to-agent transfer(based on target agent type)</li>
                      <li>• +25✖️pickup_count: Package picked up from shelf</li>
                    </ul>
                  </div>
                  <div>
                    <h5 className="font-medium mb-2 mt-2 text-red-600">Penalties</h5>
                    <ul className="text-sm text-gray-700 space-y-1">
                      <li>• -0.1: Invalid area access (transfer region violation)</li>
                      <li>• -3: Collision with solid materials(shelves,agents,etc.)</li>
                      <li>• -110✖️pickup_count: Fool for reward by picking up package in receiving region</li>
                      <li>• -0.1: Each time step (encourages efficiency)</li>
                    </ul>
                  </div>
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
