import dynamic from 'next/dynamic';

const CanvasWrapper = dynamic(() => import('../components/environment_setup'), {
    ssr: false,
});

interface WarehousePageProps{
    sessionId: string
    isTraining: boolean
    dimension: string
    envReady: boolean
}

export default function WarehousePage({sessionId,isTraining,dimension,envReady}:WarehousePageProps) {
    return <div className="w-full h-screen"><CanvasWrapper sessionId={sessionId} isTraining={isTraining} dimension={dimension} envReady={envReady}/></div>;
}