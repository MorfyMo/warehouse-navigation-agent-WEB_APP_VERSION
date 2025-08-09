"use client"
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Github, RotateCcw, Save } from "lucide-react"
import { useEffect, useState } from "react"
// import CanvasWrapper from "@/components/environment_setup"

type CodeViewerProps={
  sessionId: string | null;
};

// {sessionId}:CodeViewerProps
export function CodeViewerModal() {
  const [open, setOpen] = useState(false)

  // Sample code for each module - you can replace these with actual code
  const [envCode, setEnvCode] = useState()
  const [modCode, setModCode] = useState("")
//   const initial_layout=`(
// "1111111111111111111111111111111111111111111111111111111111111111111111\n"
// "1000002000000000000000000000008000000000000000000000000000000000000001\n"
// "1000000000000000000000000000000000000000000000000000000000000000000001\n"
// "1000311300033333333333333333333333333333333000051111111111111111150001\n"
// "1000311300011111111111111111111111111111111000011000000000000000110001\n"
// "1000311300011111111111111111111111111111111000011000000000000000110001\n"
// "1000311300033333333333333333333333333333333000011000000000000011110001\n"
// "1000311300000008000000000020000000000000000000011000000000000111110001\n"
// "1000311300000000000000000000000000000000000000011000000000000111110001\n"
// "1000311300033333333333333333333333333333333000011000000000000111110001\n"
// "1000311300011111111111111111111111111111111000011000000000000011110001\n"
// "1000311300011111111111111111111111111111111000011000000000000000110001\n"
// "1000311300033333333333333333333333333333333000051100001111111111150001\n"
// "1000000000000000000000000000020000000000000000000000000000000000000001\n"
// "1000000000000000000000000000000000000000000000000000000000000000000001\n"
// "1000444444444444444444444444444400000000000000000000000000000000000001\n"
// "1000444444444444444444444444444444444444444444444444444444444440000001\n"
// "1000444444444444444444444444444444444444444444444444444444444440000001\n"
// "1111666666666666666666966666666666666666666666666666666666666661111111\n")`
  const initial_layout=`["1111111111111111111111111111111111111111111111111111111111111111111111",
"1000002000000000000000000000008000000000000000000000000000000000000001",
"1000000000000000000000000000000000000000000000000000000000000000000001",
"1000311300033333333333333333333333333333333000051111111111111111150001",
"1000311300011111111111111111111111111111111000011000000000000000110001",
"1000311300011111111111111111111111111111111000011000000000000000110001",
"1000311300033333333333333333333333333333333000011000000000000011110001",
"1000311300000008000000000020000000000000000000011000000000000111110001",
"1000311300000000000000000000000000000000000000011000000000000111110001",
"1000311300033333333333333333333333333333333000011000000000000111110001",
"1000311300011111111111111111111111111111111000011000000000000011110001",
"1000311300011111111111111111111111111111111000011000000000000000110001",
"1000311300033333333333333333333333333333333000051100001111111111150001",
"1000000000000000000000000000020000000000000000000000000000000000000001",
"1000000000000000000000000000000000000000000000000000000000000000000001",
"1000444444444444444444444444444400000000000000000000000000000000000001",
"1000444444444444444444444444444444444444444444444444444444444440000001",
"1000444444444444444444444444444444444444444444444444444444444440000001",
"1111666666666666666666966666666666666666666666666666666666666661111111"]`

  useEffect(() => {
    setModCode(initial_layout);
  }, []);

  const resetCode = (module: string) => {
    // if (module === "layout") {
    //   useEffect(() => {
    //       setModCode(initial_layout);
    //     }, []);
    // } else if (module === "modify") {
    if(module=="modify"){
      // setModCode(dqnCode.split("\n").slice(0, 10).join("\n") + "\n# ... (original code)")
      setModCode(initial_layout)
    }
  }

  // const saveCode = () => {
  //   // In a real implementation, this would save the modified code
  //   alert("Code saved! (In a real implementation, this would save to your backend)")
  // }
  const saveCode = async () => {
    // const response = await fetch(`/api/layout_modification?session_id=${sessionId}`, {
    const response = await fetch(`${API_BASE_URL}/api/layout_modification`,{
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({layout: modCode})
    });

    if (response.ok) {
      alert("Code saved and sent to backend!");
    } else {
      alert("Failed to save code.");
    }
}


  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <Github className="h-4 w-4 mr-2" />
          Custom Agents
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-6xl max-h-[90vh]">
        <DialogHeader>
          <DialogTitle>Agents in Warehouse</DialogTitle>
          <DialogDescription>View and modify the Environment - place the agents you like in warehouse :)</DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="layout" className="w-full">
          {/* <TabsList className="grid w-full grid-cols-3"> */}
          <TabsList className="w-full flex">
            <TabsTrigger value="layout" className="w-1/2 justify-center">Layout Description</TabsTrigger>
            <TabsTrigger value="modify" className="w-1/2 justify-center">Modify Agents</TabsTrigger>
          </TabsList>

          <TabsContent value="layout" className="space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-semibold">Environment Layout</h3>
              {/* <div className="space-x-2">
                <Button variant="outline" size="sm" onClick={() => resetCode("env")}>
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Reset
                </Button>
                <Button variant="outline" size="sm" onClick={saveCode}>
                  <Save className="h-4 w-4 mr-2" />
                  Save
                </Button>
              </div> */}
            </div>
            <ScrollArea className="h-[60vh] w-full">
              {/* <Textarea
                value={envCode}
                onChange={(e) => setEnvCode(e.target.value)}
                className="min-h-[60vh] font-mono text-sm"
                placeholder="Environment module code..."
              /> */}
              <h3>
                Our warehouse can be comprehended as a matrix like this:
                <br></br>
                <br></br>
                <img src="/warehouse.png" alt="warehouse" className="w-[36rem] mx-auto" />
                <br></br>
                As can see from this image, there are numbers from 0 to 6 on it.
                <br></br>
                Each distinct number(integer) maps to a distinct component in this warehouse(explained in "Environment Details") like 1 to 1 function .
                <br></br>
                <br></br>
                The integers can be translated into warehouse components as the following:
                <br></br>
                <br></br>
                0: empty ground
                <br></br>
                1: solid(wall & shelves)
                <br></br>
                3: package
                <br></br>
                4: receiving region
                <br></br>
                5*: corner of the control room
                <br></br>
                6: transfer region
                <br></br>
                <br></br>
                You might notice that there are numbers missing from the sequence and the above image -- ' 2 '.
                <br></br>
                In fact ' 2 ' is not the only thing missing from the complete figure. From the above introduction we can 
                only see the hardwares of this warehouse -- what it is made of -- but we don't have the "robots"(the agents) in it.
                <br></br>
                <br></br>
                <h3 className="font-semibold">
                  And these are what you can add in this warehouse!
                </h3>
                As mentioned in the "Environment Detail" part, we have 3 types of agents, and their correspondence in warehouse is this:
                <br></br>
                <br></br>
                2: normal agent
                <br></br>
                8: helper agent
                <br></br>
                9: collector agent
                <br></br>
                <br></br>
                You can remove a random empty ground position on the layout and insert the agent you want onto that position and see what would happen :)
                <br></br>
                <br></br>
                <h3 className="font-semibold">
                  No Modification that exceeds the current dimension can be allowed!
                </h3>
              </h3>
            </ScrollArea>
          </TabsContent>

          <TabsContent value="modify" className="space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-semibold">Layout Modification</h3>
              <div className="space-x-2">
                <Button variant="outline" size="sm" onClick={() => resetCode("modify")}>
                  {/* <CanvasWrapper layout={modCode} /> */}
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Reset
                </Button>
                <Button variant="outline" size="sm" onClick={saveCode}>
                  <Save className="h-4 w-4 mr-2" />
                  Save
                </Button>
              </div>
            </div>
              <ScrollArea className="h-[60vh] w-full">
                <h3>
                  <p>
                    Modify number in the layout. Replace "0" for agent. Click "Save" to Save.
                    <br></br>
                    <span className="underline font-semibold"> Do No Modify the Dimension!</span>
                  </p>
                  <br></br>
                  <br></br>
                </h3>
                {/* <div className="flex justify-center min-h-[35rem] items-center"> */}
                {/* <div className="flex justify-center w-full"> */}
                {/* space-x-number controls the distance between two blocks*/}
                  <div className="flex items-start space-x-1">
                    <div className="text-sm ml-4 text-gray-500 leading-relaxed whitespace-nowrap">
                      <strong>Remember:</strong>
                      <br></br>
                      2: normal agent
                      <br></br>
                      8: helper agent
                      <br></br>
                      9: collector agent
                      <br></br>

                    </div>
                    <br></br>
                    <br></br>
                    <div className="flex justify-center w-full">
                      <div className="border-4 border-black p-8 min-h-[35rem] items-center rounded-md">
                        <textarea
                          value={modCode}
                          onChange={(e) => setModCode(e.target.value)}
                          className="w-[40rem] min-h-[100vh] font-mono text-sm"
                          placeholder="Initial layout"
                        />
                      </div>
                    </div>
                  </div>
                {/* </div> */}
              </ScrollArea>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}
