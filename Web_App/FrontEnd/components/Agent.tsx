// import React from 'react';

export function Wall({ position, id }: { position: [number, number, number]; id: string }) {
    // const color = type === 'solid' ? '#333' : '#888';
    // return (
    // <mesh position={position}>
    //     <boxGeometry args={[1, 1, 1]} />
    //     <meshStandardMaterial color={color} />
    // </mesh>
    // );
    return (
    <mesh position={position}>
        <boxGeometry args={[1, 4, 1]} />
        <meshStandardMaterial color={'#1C1C1C'} />
    </mesh>
    );

}


export function Transfer({ position, id }: { position: [number, number, number]; id: string }) {
    // const color = type === 'solid' ? '#333' : '#888';
    // return (
    // <mesh position={position}>
    //     <boxGeometry args={[1, 1, 1]} />
    //     <meshStandardMaterial color={color} />
    // </mesh>
    // );
    return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={position}>
        <planeGeometry args={[1, 1]} />
        <meshStandardMaterial color={'#9BAABF'} />
    </mesh>
    );
}

export function Empty({ position, id }: { position: [number, number, number]; id: string }) {
    return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={position}>
        <planeGeometry args={[1, 1]} />
        <meshStandardMaterial color={'#BBBBBB'} />
    </mesh>
    );

}

export function Receiving({ position, id }: { position: [number, number, number]; id: string }) {
    return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={position}>
        <planeGeometry args={[1, 1]} />
        <meshStandardMaterial color={'#E5D99B'} />
    </mesh>
    );

}
// export default function Agent({ type, position }: { type: string; position: [number, number, number] }) {
//     const colors: Record<string, string> = {
//     agent: "red",
//     trans_agent: "blue",
//     recei_agent: "purple"
//     };

//     return (
//     <mesh position={position}>
//         <sphereGeometry args={[0.3, 32, 32]} />
//         <meshStandardMaterial color={colors[type]} />
//     </mesh>
//     );
// }

export function Agent({ position, id }: { position: [number, number, number]; id: string }) {
    return (
    <group position={position}>
        <mesh position={[0,0.9,0]}>
            <boxGeometry args={[0.8, 0.45, 0.5]} />
            <meshStandardMaterial color={'#FD251F'} />
        </mesh>
        <mesh position={[0,0.5,0]}>
            <boxGeometry args={[0.55,1,0.4]} />
            <meshStandardMaterial color={'#94928f'} />
        </mesh>
        <mesh position={[-0.3,0.3,0]}>
            <boxGeometry args={[0.15,0.65,0.23]} />
            <meshStandardMaterial color={'#f2eee8'} />
        </mesh>
        <mesh position={[0.3,0.3,0]}>
            <boxGeometry args={[0.15,0.65,0.23]} />
            <meshStandardMaterial color={'#f2eee8'} />
        </mesh>
        <mesh position={[0,0.01,0]}>
            <boxGeometry args={[1,0.2,0.6]} />
            <meshStandardMaterial color={'#d4cec6'} />
        </mesh>
    </group>
    );
}

export function Reci_Agent({ position, id }: { position: [number, number, number]; id: string }) {
    return (
    <group position={position}>
        <mesh position={[0,0.9,0]}>
            <boxGeometry args={[0.8, 0.45, 0.5]} />
            <meshStandardMaterial color={'#1919A6'} />
        </mesh>
        <mesh position={[0,0.5,0]}>
            <boxGeometry args={[0.55,1,0.4]} />
            <meshStandardMaterial color={'#94928f'} />
        </mesh>
        <mesh position={[-0.3,0.3,0]}>
            <boxGeometry args={[0.15,0.65,0.23]} />
            <meshStandardMaterial color={'#f2eee8'} />
        </mesh>
        <mesh position={[0.3,0.3,0]}>
            <boxGeometry args={[0.15,0.65,0.23]} />
            <meshStandardMaterial color={'#f2eee8'} />
        </mesh>
        <mesh position={[0,0.01,0]}>
            <boxGeometry args={[1,0.2,0.6]} />
            <meshStandardMaterial color={'#d4cec6'} />
        </mesh>
    </group>
    );
}

export function Trans_Agent({ position, id }: { position: [number, number, number]; id: string }) {
    return (
    <group position={position}>
        <mesh position={[0,0.9,0]}>
            <boxGeometry args={[0.8, 0.45, 0.5]} />
            <meshStandardMaterial color={'#049CD8'} />
        </mesh>
        <mesh position={[0,0.5,0]}>
            <boxGeometry args={[0.55,1,0.4]} />
            <meshStandardMaterial color={'#94928f'} />
        </mesh>
        <mesh position={[-0.3,0.3,0]}>
            <boxGeometry args={[0.15,0.65,0.23]} />
            <meshStandardMaterial color={'#f2eee8'} />
        </mesh>
        <mesh position={[0.3,0.3,0]}>
            <boxGeometry args={[0.15,0.65,0.23]} />
            <meshStandardMaterial color={'#f2eee8'} />
        </mesh>
        <mesh position={[0,0.01,0]}>
            <boxGeometry args={[1,0.2,0.6]} />
            <meshStandardMaterial color={'#d4cec6'} />
        </mesh>
    </group>

    );
}


export function Package({ position, id }: { position: [number, number, number]; id: string }) {
    return (
    <mesh position={position}>
        <boxGeometry args={[1.7, 2, 1.7]} />
        <meshStandardMaterial color={'#d4af7c'} />
    </mesh>
    );
}
