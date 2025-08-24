//now we import the components of different items(agents, packages, etc) to this file
// import {Box} from '@react-three/drei';
// import React from 'react';
import { Agent, Empty, Package, Receiving, Reci_Agent, Trans_Agent, Transfer, Wall } from './Agent';
// const warehouse_dimension=[19,7,70] //this means: width(x: left-right), height(y: up-down), depth(z: front-back)

const objects={
    empty:0,
    wall:1,
    agent:2,
    package:3,
    receiving:4,
    control:5,
    transfer:6,
    updated_receiving:7,
    trans_agent:8,
    recei_agent:9
}

const colors={
    gray:0,
    black:1,
    red:2,
    brown:3,
    yellow:4,
    black_control:5,
    sky:6,
    yellow_update:7,
    blue:8,
    sea:9
}

export default function WarehouseScene({ grid }: { grid: number[][] }) {
    return (
    <group>
        {grid.map((row, z) =>
            row.map((cell, x) => {
                const pos_agent: [number, number, number] = [x-35, 0.07, z-10];
                const pos_solid: [number, number, number] = [x-35, 2, z-10];
                const pos_plane: [number, number, number] = [x-35, 0.01, z-10];
                const pos_pkg: [number, number, number] = [x-35, 0.5, z-10];
                switch (cell) {
                case objects.agent:
                    return <Agent key={`agent-${x}-${z}`} position={pos_agent} id={`a-${x}-${z}`} />;
                case objects.recei_agent:
                    return <Reci_Agent key={`helper-${x}-${z}`} position={pos_agent} id={`h-${x}-${z}`} />;
                case objects.trans_agent:
                    return <Trans_Agent key={`collector-${x}-${z}`} position={pos_agent} id={`c-${x}-${z}`} />;
                case objects.control:
                    return <Wall key={`wall-${x}-${z}`} position={pos_solid} id={`w-${x}-${z}`} />;
                case objects.transfer:
                    return <Transfer key={`transfer-${x}-${z}`} position={pos_plane} id={`t-${x}-${z}`} />;
                case objects.receiving:
                    return <Receiving key={`receiving-${x}-${z}`} position={pos_plane} id={`r-${x}-${z}`} />;
                case objects.updated_receiving:
                    return <Receiving key={`receiving-${x}-${z}`} position={pos_plane} id={`r-${x}-${z}`} />;
                case objects.empty:
                    return <Empty key={`empty-${x}-${z}`} position={pos_plane} id={`e-${x}-${z}`} />;
                case objects.wall:
                    return <Wall key={`wall-${x}-${z}`} position={pos_solid} id={`w-${x}-${z}`} />;
                case objects.package:
                    return <Package key={`pkg-${x}-${z}`} position={pos_pkg} id={`p-${x}-${z}`} />;
                default:
                    return null;
                }
            })
        )}
    </group>
    );
}