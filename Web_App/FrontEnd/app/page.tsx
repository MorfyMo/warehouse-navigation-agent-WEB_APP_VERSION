"use client"

import { AlgorithmComparison } from "@/components/algorithm-comparison"
import { CodeViewerModal } from "@/components/code-viewer-modal"
import { DocumentationModal } from "@/components/documentation-modal"
// import { EnvironmentVisualization } from "@/components/environment-visualization"
// we remove this line because we are using rendering from the environment module
import Comments from "@/components/Messages"
import { MetricsChart } from "@/components/metrics-chart"
import { ThemeToggle } from "@/components/theme-toggle"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
// import { lastStatsDqn, lastStatsPpo } from "@/components/ui/stats-store"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { WarehouseInfo } from "@/components/warehouse-info"
import { WarehouseRender } from "@/components/warehouse_render"
import { useWS } from "@/hooks/useWS"
import { waitForEnv } from "@/lib/api"
import WarehousePage from "./threeD"

// import { WarehouseVisualization } from "@/components/warehouse-visualization"
import { api, wsPaths, type TrainingConfig } from "@/lib/api"
import { BarChart3, Brain, Pause, Play, RotateCcw, Settings, Warehouse, Zap } from "lucide-react"
import { useTheme } from "next-themes"
import { useCallback, useEffect, useRef, useState } from "react"

// this is for the result(Performance) showing part
// export let lastStatsDqn={average_reward: 0.0, time: "0.0s"}
// export let lastStatsPpo = {normalized_reward: 0.0, time: "0.0s"}

  // Initialized refs with default values
  // export const lastStatsDqn= useRef({
  //   average_reward: 0.0,
  //   time: "0.0s",
  // });
  // lastStatsDqn.current = {
  //   average_reward: 0.0,
  //   time: "0.0s",
  // };

  // export const lastStatsPpo= useRef({
  //   normalized_reward: 0.0,
  //   time: "0.0s",
  // });
  // lastStatsPpo.current = {
  //   normalized_reward: 0.0,
  //   time: "0.0s",
  // };
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080"
const NEXT_PUBLIC_WS_URL=process.env.NEXT_PUBLIC_WS_URL ?? "wss://warehouse-rl-api.fly.dev"

export const dynamic = 'force-dynamic'; // disables static optimization
// export const revalidate = 0;

export default function RLShowcase() {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<"dqn" | "ppo">("dqn")
  const [selectedDimension, setSelectedDimension] = useState<"2D" | "3D">("2D")
  // const [selectedEnvironment, setSelectedEnvironment] = useState<"navigation" | "warehouse">("navigation")
  const [selectedEnvironment, setSelectedEnvironment] = useState<"warehouse">("warehouse")
  const [isTraining, setIsTraining] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [envReady, setEnvReady] = useState(false)
  const [wantStart, setWantStart] = useState(false);
  // this constant ensures that the start button can only be triggered once even if the user click on the button twice or more:
  const [isStarting, setIsStarting] = useState(false);
  // const [plotURL, setplotURL] = useState<string | null | undefined>(null)

  const lastStatsDqn = useRef({average_reward: 0.0, time:"0.0s", delivered: 0})
  const lastStatsPpo = useRef({normalized_reward:0.0, time:"0.0s", delivered: 0})
  const [dqnData, setDqnData]=useState([{average_reward: 0.0, time:"0.0s", delivered: 0}]);
  const [ppoData, setPpoData] = useState([{normalized_reward: 0.0, time:"0.0s", delivered: 0}])

  const [trainingAlgo, setTrainingAlgo] = useState("DQN")
  const [episodes, setEpisodes] = useState([1000]) //this function is used to store and track data, react to interaction and keep UI sync with internal data
  //this "[1000]" means creating a list with 1 element - 1000
  //"1000" is the default selection(the detailed upper and lower limit would be defined in 'Slider')
  const [learningRate, setLearningRate] = useState([0.001])
  const [totalPackages, setTotalPackages] = useState([146])
  const [mounted, setMounted] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  // const [socket2d, setSocket2d] = useState<WebSocket | null>(null);
  // const [socket3d, setSocket3d] = useState<WebSocket | null>(null);

  // this below is for the original wsConnections(before using "useWS"):
  // const [wsConnection, setWsConnection] = useState<TrainingWebSocket | null>(null)
  const { theme } = useTheme()


  // Avoid hydration mismatch by only rendering theme-dependent elements after mount
  useEffect(() => {
    setMounted(true)
  }, [])

  // Cleanup WebSocket on unmount(this below is the version for useEffect before using "useWS")
  // useEffect(() => {
  //   return () => {
  //     if (wsConnection) {
  //       wsConnection.disconnect()
  //     }
  //   }
  // }, [wsConnection])

  function validId(id: unknown): id is string{
      return typeof id==="string" && id!=="undefined" && id!== "null" && id.length>0;
  }

  const handlePlotMsg = useCallback((event: MessageEvent) => {
    if (typeof event.data !== "string") return;  // ignore binary
    // the below two lines are to prevent issue(after "useWS" added):
    let data: any
    try {data = JSON.parse(event.data);} catch {return;}


    if (data.type === "ping") {
      // if ever want to reply: sendPlotRef.current?.({ type:"pong", ts: Date.now() });
      return;
    }

    // if(typeof data.image_base64 === "string"){
    //   setFrame64(data.image_base64)
    // }

    if (data.current_algo === "dqn") {
      if (!data.stopped) {
        setDqnData(prev => [
          ...prev.slice(-50),
          { average_reward: data.reward, time: data.time_text, delivered: data.delivered }
        ]);
        lastStatsDqn.current = { average_reward: data.reward, time: data.time_text, delivered: data.delivered };
      } else {
        setDqnData(prev => [...prev.slice(-50), lastStatsDqn.current]);
      }
    }

    if (data.current_algo === "ppo") {
      if (!data.stopped) {
        setPpoData(prev => [
          ...prev.slice(-50),
          { normalized_reward: data.reward, time: data.time_text, delivered: data.delivered }
        ]);
        lastStatsPpo.current = { normalized_reward: data.reward, time: data.time_text, delivered: data.delivered };
      } else {
        setPpoData(prev => [...prev.slice(-50), lastStatsPpo.current]);
      }
    }
  }, [setDqnData, setPpoData]);

  const handleTrainMsg = useCallback((evt: MessageEvent) => {
    if (typeof evt.data !== "string") return;
    let data: any; try { data = JSON.parse(evt.data) } catch { return; }

    if (data.type === "render" && typeof data.progress === "number") {
      setTrainingProgress(data.progress);
    } else if (data.type === "training_complete") {
      setIsTraining(false);
      setTrainingProgress(100);
    } else if (data.type === "error") {
      setIsTraining(false);
    }
  }, []);

  const trainPath = validId(sessionId) && isTraining
    ? wsPaths.training(sessionId) : undefined;

  const plotPath =
    validId(sessionId) && isTraining && envReady
      ? (selectedDimension === "2D" ? wsPaths.plot2d(sessionId) : wsPaths.plot3d(sessionId))
      : undefined;

  const { status: trainStatus, send: sendTrain } = useWS({
    path: trainPath,
    onMessage: handleTrainMsg,
    throttleMs: 16, // ~60fps for real-time training updates
  });

  const { status: plotStatus, send: sendPlot } = useWS({
    path: plotPath,
    onMessage: handlePlotMsg,
    throttleMs: 16, // ~60fps for smooth visualization
    // binaryType: "arraybuffer", // enable later if you switch to binary frames
  });

  // This below is the original version before using "useWS":
  // useEffect(() => {
  //   if(!sessionId||!isTraining) return;

  //   const handleMessage = (event: MessageEvent) => {
  //     const data = JSON.parse(event.data);
  //     if(data.type==="ping"){
  //       // socket.send(JSON.stringify({ type: "pong", ts: Date.now() }));
  //       return;
  //     }

  //     if(data.current_algo === "dqn"){
  //       if(!data.stopped){
  //         setDqnData(prev => [
  //           ...prev.slice(-50),
  //           {average_reward: data.reward, time: data.time_text, delivered: data.delivered}
  //         ]);
  //         lastStatsDqn.current = {average_reward: data.reward, time: data.time_text, delivered: data.delivered};
  //       }else{
  //         setDqnData(prev => [...prev.slice(-50), lastStatsDqn.current]);
  //       }
  //     }
  //     if(data.current_algo === "ppo"){
  //       if(!data.stopped){
  //         setPpoData(prev => [
  //           ...prev.slice(-50),
  //           {normalized_reward: data.reward, time: data.time_text, delivered: data.delivered}
  //         ]);
  //         lastStatsPpo.current = {normalized_reward: data.reward, time: data.time_text, delivered: data.delivered}
  //       }else{
  //         setPpoData(prev => [...prev.slice(-50), lastStatsPpo.current]);
  //       }
  //     }
  //   };

  //   let socket: WebSocket | null = null

  //   if(selectedDimension==="2D"){
  //     socket = new WebSocket(`${NEXT_PUBLIC_WS_URL}/ws/plot/${sessionId}`);
  //     socket.onmessage = handleMessage;
  //   } else {
  //     socket = new WebSocket(`${NEXT_PUBLIC_WS_URL}/ws/plot3d/${sessionId}`);
  //     socket.onmessage = handleMessage;
  //   }

  //   return () => {
  //     if (socket) socket.close()
  //   };

  // },[selectedAlgorithm, sessionId, isTraining, selectedDimension]);

  // these below 2 blocks(about plot) are removed to make room for the socket inside warehouse_render function:
  const sendPlotRef = useRef<typeof sendPlot | null>(null);

  useEffect(() => {
    sendPlotRef.current = sendPlot;
    return () => { sendPlotRef.current = null; };
  }, [sendPlot]);

  // useEffect(() => {
  //   if (!wantStart) return;
  //   if (trainStatus === "open") {
  //     sendTrain({ type: "ready" });
  //     setWantStart(false); // prevent re-sending
  //   }
  // }, [wantStart, trainStatus, sendTrain]);

  const trainKickoff = useRef(false);
  useEffect(()=>{trainKickoff.current=false},[trainPath]);
  useEffect(()=>{
    if(trainStatus === "open" && trainPath && !trainKickoff.current){
      sendTrain({type:"ready"})
      trainKickoff.current=true
    }
  }, [trainStatus, trainPath, sendTrain]);

  // this block is removed becaue we need to make room for the warehouse_render version of 2dplot showing:
  const plotKickoff = useRef(false);
  useEffect(()=>{plotKickoff.current=false},[plotPath]);
  useEffect(()=>{
    if(plotStatus==="open" && plotPath && !plotKickoff.current){
      sendPlot({type:"ready"})
      plotKickoff.current=true
    }
  },[plotStatus, plotPath, sendPlot]);

  // this checks whether the environment is ready
  useEffect(()=>{
    let cancelled = false;
    setEnvReady(false)
      if(!validId(sessionId) || !isTraining){
        return;
      };

    (async() =>{

      try{
        await waitForEnv(sessionId,{timeoutMs: 30_000, intervalMs: 500});
        if(!cancelled) setEnvReady(true);

      } catch(e) {
        if(!cancelled) setEnvReady(false);
        console.warn("env_init waiter:",e);
      }
    })();
    return () => { cancelled = true;};

  }, [sessionId, isTraining]);

  const simulateTraining = () => {
    setIsTraining(true)
    setTrainingProgress(0)

    const interval = setInterval(() => {
      setTrainingProgress((prev) => {
        if (prev >= 100) {
          setIsTraining(false)
          clearInterval(interval)
          return 100
        }
        return prev + 2
      })
    }, 100)
  }

  const handleStartTraining = async () => {
    // this condition ensures that we are not double triggering the start training button
    if(isTraining || isStarting) return;
    setIsStarting(true);
    try {
      const config: TrainingConfig = {
        algorithm: selectedAlgorithm,
        episodes: episodes[0], //this means select the first(and also only) element from the list "episodes"
        learning_rate: learningRate[0],
        environment: selectedEnvironment,
        //if our selected environment is warehouse, we include the warehouse config in the final config; otherwise, include just nothing
        ...(selectedEnvironment === "warehouse" && {
          warehouse_config: {
            total_packages: totalPackages[0],
            agent_types: ["normal", "helper", "collector"],
          },
        }),
      }

      const result = await api.startTraining(config)
      const algo=config.algorithm
      setTrainingAlgo(algo)

      // if (result.success) {
      if(!result?.success||!result?.session_id) throw new Error("No session_id now");
        setSessionId(result.session_id)
        setIsTraining(true)
        setTrainingProgress(0)
        setWantStart(true);
        setIsStarting(true);

    } catch (error) {
      console.error("Failed to start training:", error)
      simulateTraining()
    } finally{
      setIsStarting(false)
    }
  }

  const handleReset = async () => {
    // if (sessionId && wsConnection) {
    if(sessionId && validId(sessionId)){
      try {
        await api.stopTraining(sessionId)
        // wsConnection.disconnect()
        sendTrain?.({type:"stop"})

      } catch (error) {
        console.error("Failed to stop training:", error)
      }
    }

    setIsTraining(false)
    setTrainingProgress(0)
    setSessionId(null)
    // setWsConnection(null)
  }

  // the const of lastDqn and lastPpo doesn't change
  const lastDqn = dqnData[dqnData.length - 1] ?? dqnData[0];
  const lastPpo = ppoData[ppoData.length - 1] ?? ppoData[0];

  // Determine if we're in light or dark mode
  const isLightMode = mounted && theme === "light"

  return (
    <div
      className={
        isLightMode
          ? "min-h-screen bg-gradient-to-br from-slate-50 to-slate-100"
          : "min-h-screen bg-gradient-to-br from-background to-background/80"
      }
    >
      {/* Header */}
      <header
        className={
          isLightMode
            ? "border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50"
            : "border-b bg-background/80 backdrop-blur-sm sticky top-0 z-50"
        }
      >
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Brain className={isLightMode ? "h-8 w-8 text-blue-600" : "h-8 w-8 text-primary"} />
              <h1 className={isLightMode ? "text-2xl font-bold text-gray-900" : "text-2xl font-bold"}>
                Warehouse Agents Navigation
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <ThemeToggle />
              <CodeViewerModal />
              {/* sessionId={sessionId}  */}
              <DocumentationModal />
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-16 px-4">
        <div className="container mx-auto text-center">
          <h2 className={isLightMode ? "text-4xl font-bold text-gray-900 mb-4" : "text-4xl font-bold mb-4"}>
            Warehouse Agents Navigation
          </h2>
          <p
            className={
              isLightMode
                ? "text-xl text-gray-600 mb-8 max-w-3xl mx-auto"
                : "text-xl text-muted-foreground mb-8 max-w-3xl mx-auto"
            }
          >
            Comparison of DQN and PPO algorithms in a warehouse. Switch between algorithms, adjust parameters, and
            visualize training results in real-time.
          </p>
          <div className="flex justify-center space-x-4 mb-8">
            <Badge variant="secondary" className="px-4 py-2">
              <Zap className="h-4 w-4 mr-2" />
              TensorFlow
            </Badge>
            <Badge variant="secondary" className="px-4 py-2">
              <Brain className="h-4 w-4 mr-2" />
              Deep Q-Network
            </Badge>
            <Badge variant="secondary" className="px-4 py-2">
              <BarChart3 className="h-4 w-4 mr-2" />
              PPO Algorithm
            </Badge>
          </div>
        </div>
      </section>

      {/* {selectedDimension==="2D" &&
        <PerformanceChartDqn
          algorithm={selectedAlgorithm}
          isTraining={isTraining}
          sessionId={sessionId}
          mode={selectedDimension}
        />
      }
      {selectedDimension==="3D" &&
        <PerformanceChartDqn3d
          algorithm={selectedAlgorithm}
          isTraining={isTraining}
          sessionId={sessionId}
          mode={selectedDimension}
        />
      }
      {selectedDimension==="2D" &&
        <PerformanceChartPpo
          algorithm={selectedAlgorithm}
          isTraining={isTraining}
          sessionId={sessionId}
          mode={selectedDimension}
        />
      }
      {selectedDimension==="3D" &&
        <PerformanceChartPpo3d
          algorithm={selectedAlgorithm}
          isTraining={isTraining}
          sessionId={sessionId}
          mode={selectedDimension}
        />
      } */}

      {/* Main Content */}
      <div className="container mx-auto px-4 pb-16">
        <Tabs defaultValue="demo" className="space-y-8">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger className="flex-1" value="demo">Warehouse Navigation</TabsTrigger>
            <TabsTrigger className="flex-1" value="comparison">Algorithm Comparison</TabsTrigger>
            <TabsTrigger className="flex-1" value="results">Training Results</TabsTrigger>
            <TabsTrigger className="flex-1" value="environment">Environment Details</TabsTrigger>
            <TabsTrigger className="flex-1" value="suggestions">Friendly Suggestions</TabsTrigger>
          </TabsList>

          {/* Interactive Demo Tab */}
          <TabsContent value="demo" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Control Panel */}
              <Card className="lg:col-span-1">
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Settings className="h-5 w-5 mr-2" />
                    Training Controls
                  </CardTitle>
                  <CardDescription>Configure and run your RL algorithms</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Environment</label>
                    <Select
                      value={selectedEnvironment}
                      // onValueChange={(value: "navigation" | "warehouse") => setSelectedEnvironment(value)}
                      onValueChange={(value:"warehouse")=>setSelectedEnvironment(value)}
                    >
                      <SelectTrigger>
                        <SelectValue/>
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="warehouse">
                          <div className="flex items-center">
                            <Warehouse className="h-4 w-4 mr-2" />
                            Warehouse Robot Navigation
                          </div>
                        </SelectItem>
                        {/* <SelectItem value="navigation">
                          <div className="flex items-center">
                            <Navigation className="h-4 w-4 mr-2" />
                            Navigation Environment
                          </div>
                        </SelectItem> */}

                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <label className="text-sm font-medium mb-2 block">Algorithm</label>
                    <Select
                      value={selectedAlgorithm}
                      onValueChange={(value: "dqn" | "ppo") => setSelectedAlgorithm(value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="dqn">Deep Q-Network (DQN)</SelectItem>
                        <SelectItem value="ppo">Proximal Policy Optimization (PPO)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div>
                    <label className="text-sm font-medium mb-2 block">Dimension Selection</label>
                    <Select
                      value={selectedDimension}
                      onValueChange={(value: "2D" | "3D") => setSelectedDimension(value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="2D">2D View (with value)</SelectItem>
                        <SelectItem value="3D">3D View</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  {/* <WarehousePage /> */}

                  <div>
                    <label className="text-sm font-medium mb-2 block">Episodes: {episodes[0]}
                    </label>
                    <Slider
                      value={episodes}
                      onValueChange={setEpisodes}
                      // onValueChange={(newVal)=>{
                      //   setEpisodes(newVal)

                      //   //now we need to send the selected episode to the backend
                      //   const selectedEpisode = newVal[0]
                      //   if(sessionId){
                      //     api.setRenderEpisode(sessionId,selectedEpisode)
                      //   }
                      // }}
                      max={5000}
                      min={50}
                      step={50}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium mb-2 block">Learning Rate: {learningRate[0]}</label>
                    <Slider
                      value={learningRate}
                      onValueChange={setLearningRate}
                      max={0.01}
                      min={0.0001}
                      step={0.0001}
                      className="w-full"
                    />
                  </div>

                  {selectedEnvironment === "warehouse" && (
                    <div>
                      <label className="text-sm font-medium mb-2 block">Package Deliver Goal: {totalPackages[0]}</label>
                      <Slider
                        value={totalPackages}
                        onValueChange={setTotalPackages}
                        max={146}
                        min={10}
                        step={5}
                        className="w-full"
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        Default: 146 packages (all the packages currently exist in this warehouse)
                      </p>
                    </div>
                  )}

                  <div className="space-y-3">
                    <div className="flex space-x-2">
                      <Button onClick={handleStartTraining} disabled={isTraining||isStarting} className="flex-1">
                        {isTraining ? (
                          <>
                            <Pause className="h-4 w-4 mr-2" />
                            Training...
                          </>
                        ) : (
                          <>
                            <Play className="h-4 w-4 mr-2" />
                            Start Training
                          </>
                        )}
                      </Button>
                      <Button variant="outline" onClick={handleReset}>
                        <RotateCcw className="h-4 w-4" />
                      </Button>
                    </div>

                    {(isTraining || envReady) && (
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Episode Progress</span>
                          <span>{trainingProgress}%</span>
                        </div>
                        <Progress value={trainingProgress} className="w-full" />
                      </div>
                    )}
                  </div>

                  <div className="pt-4 border-t">
                    <h4 className="font-medium mb-2">
                      Current: {selectedAlgorithm.toUpperCase()} -{" "}
                      {selectedEnvironment === "warehouse" ? "Warehouse" : "Navigation"}
                    </h4>
                    <p className={isLightMode ? "text-sm text-gray-600" : "text-sm text-muted-foreground"}>
                      {selectedAlgorithm === "dqn"
                        ? "Deep Q-Network uses experience replay and target networks for stable learning."
                        : "PPO uses clipped surrogate objective for stable policy gradient updates."}
                    </p>
                  </div>
                </CardContent>
              </Card>

              {/* Environment Visualization */}
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle>Environment Visualization</CardTitle>
                  <CardDescription>
                    {selectedEnvironment === "warehouse"
                      ? "Real-time view of warehouse robots managing packages"
                      : "Real-time view of the agent interacting with the environment"}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {sessionId && selectedDimension === "2D" &&(
                      <WarehouseRender sessionId={sessionId} isTraining={isTraining} mode={selectedDimension} envReady={envReady}/>
                    )}
                  {sessionId && selectedDimension === "3D" &&(
                      <WarehousePage sessionId={sessionId} isTraining={isTraining} dimension={selectedDimension} envReady={envReady}/>
                    )
                  }
                </CardContent>
              </Card>
            </div>

            {/* Warehouse Info */}
            {/* {selectedEnvironment === "warehouse" && <WarehouseInfo />} */}

            {/* Metrics */}
            <Card>
              <CardHeader>
                <CardTitle>Training Metrics</CardTitle>
                <CardDescription>Real-time performance metrics during training</CardDescription>
              </CardHeader>
              <CardContent>
                <MetricsChart
                  algorithm={selectedAlgorithm}
                  isTraining={isTraining}
                  progress={trainingProgress}
                  sessionId={sessionId}
                  mode={selectedDimension}
                  envReady={envReady}
                  // rewardData={rewardData}
                  // lossData={lossData}
                  // epsilonData={epsilonData}
                  // actorlossData={actorlossData}
                  // criticlossData={criticlossData}
                />
              </CardContent>
            </Card>
          </TabsContent>

          {/* Algorithm Comparison Tab */}
          <TabsContent value="comparison">
            <AlgorithmComparison />
          </TabsContent>

          {/* Training Results Tab */}
          <TabsContent value="results">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>DQN Performance</CardTitle>
                  <CardDescription>Training results for Deep Q-Network</CardDescription>
                </CardHeader>
                <CardContent>
                  {/* {selectedDimension==="2D" &&
                    <PerformanceChartDqn
                      algorithm={selectedAlgorithm}
                      isTraining={isTraining}
                      sessionId={sessionId}
                      mode={selectedDimension}
                    />
                  }
                  {selectedDimension==="3D" &&
                    <PerformanceChartDqn3d
                      algorithm={selectedAlgorithm}
                      isTraining={isTraining}
                      sessionId={sessionId}
                      mode={selectedDimension}
                    />
                  } */}
                  
                  <div className="space-y-4">
                      <div className="flex justify-between">
                        <span>Average Reward:</span>
                        <span className="font-mono">{lastDqn?.average_reward}</span>
                      </div>
                      {/* <div className="flex justify-between">
                          <span>Episodes to Convergence:</span>
                          <span className="font-mono">1,250</span>
                      </div> */}
                      <div className="flex justify-between">
                        <span>Training Time:</span>
                        <span className="font-mono">{lastDqn?.time}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Packages delivered:</span>
                        <span className="font-mono">{lastDqn?.delivered} pkg</span>
                      </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>PPO Performance</CardTitle>
                  <CardDescription>Training results for Proximal Policy Optimization</CardDescription>
                </CardHeader>
                <CardContent>
                {/* {selectedDimension==="2D" &&
                  <PerformanceChartPpo
                    algorithm={selectedAlgorithm}
                    isTraining={isTraining}
                    sessionId={sessionId}
                    mode={selectedDimension}
                  />
                }
                {selectedDimension==="3D" &&
                  <PerformanceChartPpo3d
                    algorithm={selectedAlgorithm}
                    isTraining={isTraining}
                    sessionId={sessionId}
                    mode={selectedDimension}
                  />
                } */}

                  <div className="space-y-4">
                      <div className="flex justify-between">
                          <span>Normalized Reward:</span>
                          <span className="font-mono">{lastPpo?.normalized_reward}</span>
                      </div>
                      {/* <div className="flex justify-between">
                          <span>Episodes to Convergence:</span>
                          <span className="font-mono">980</span>
                      </div> */}
                      <div className="flex justify-between">
                          <span>Training Time:</span>
                          <span className="font-mono">{lastPpo?.time}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Packages delivered:</span>
                        <span className="font-mono">{lastPpo?.delivered} pkg</span>
                      </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Environment Details Tab */}
          <TabsContent value="environment">
            {selectedEnvironment === "warehouse" ? (
              <WarehouseInfo detailed />
            ) : (
              <Card>
                <CardHeader>
                  <CardTitle>Navigation Environment Specifications</CardTitle>
                  <CardDescription>Details about the navigation environment implementation</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
                    <div>
                      <h4 className="font-bold text-lg mb-4">State Space</h4>
                      <ul className="space-y-3 text-base">
                        <li>• Discrete state space with 19×70 dimensions</li>
                        <li>• A total of 9 states:</li>
                        <li>
                          <strong>1)</strong> empty (gray): the ground the agents step on
                        </li>
                        <li>
                          <strong>2)</strong> wall (black): the walls or shelves where it is harmful if the agents crash
                          into them
                        </li>
                        <li>
                          <strong>3)</strong> receiving (yellow): where the normal agents and helper agents put down the
                          package (when a package is put onto the receiving region, it is delivered; if it is picked up
                          later by the normal agent again, it is again not delivered)
                        </li>
                        <li>
                          <strong>4)</strong> control (black): this is the major positions to tag the space of the
                          control room - which currently have no special usage
                        </li>
                        <li>
                          <strong>5)</strong> transfer (sky blue): the region where the collector agent transfer the
                          package to - to somehow reduce the burden of the receiving region and maybe to send the
                          package out for external delivery
                        </li>
                        <li>
                          <strong>6)</strong> agents: the agents that deals with the package delivery inside this
                          warehouse. There are 3 types of agents:
                        </li>
                        <li className="ml-4">
                          • normal agent (red): the agent that has the capability to pick up both the packages on the
                          receiving region and the packages on the shelves and put them down on receiving region
                        </li>
                        <li className="ml-4">
                          • helper (blue): the agent that has the capability to take package from the normal agents and
                          put it down on the receiving region - but doesn't have the capability to pickup the packages
                        </li>
                        <li className="ml-4">
                          • collector (dark blue): the agent that pick up packages on the receiving region and transmit
                          it to the transfer region - it only move on the receiving region and the transfer region, and
                          does not take packages from the other agents
                        </li>
                        <li>
                          <strong>7)</strong> package (brown): the large package, each one is a huge box - this is used
                          for agents for delivery
                        </li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-bold text-lg mb-4">Action Space</h4>
                      <ul className="space-y-3 text-base">
                        <li>• Discrete action space with 7 actions</li>
                        <li>• Do not move</li>
                        <li>• Move forward/backward</li>
                        <li>• Turn left/right</li>
                        <li>• Turn the package to the left/right side</li>
                        <li>• Action magnitude: [-1, 1]</li>
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="suggestions">
            <Card>
              <CardHeader>
                <div className="text-center">
                  <CardTitle >Friendly Messages</CardTitle>
                  plz be friendly :)
                </div>
                
              </CardHeader>
              <Comments/>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      <footer className="bg-black text-white dark:bg-white dark:text-black py-5 mt-8">
        <div className="max-w-6xl space-y-2 mx-auto px-5 text-center">
          {/* <p className="text-lg font-semibold">...</p> */}
          <p className="text-sm">
            GitHub:{' '}
            <a
              href="https://github.com/MorfyMo/Warehouse-Navigation-Agent_MorfyMo"
              target="_blank"
              rel="noopener noreferrer"
              className="underline hover: text-gray-300 dark:text-gray-700"
            >
              https://github.com/MorfyMo/Warehouse-Navigation-Agent_MorfyMo
            </a>
          </p>
          
          <p className="text-xs text-gray-400">@MorfyMo</p>
        </div>
      </footer>
    </div>
  )
}
