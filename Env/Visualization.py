"""
This is the class used to showcase the whole thing to 3D instead of 2D
note that for (x,y,z), we follows the following rule:
* x and z forms a plane on the ground, while y is the actual height from the ground(either high up or down)
1) x: the width, moving left and right
2) z: the length, moving forward and backward(replace the traditional y axis on the Cartesian coordinate)
3) y: the actual height, the distance from above the ground or below the ground
"""
import asyncio
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # type: ignore
import plotly.graph_objects as go
import numpy as np
import io
import base64
from fastapi import WebSocket, WebSocketDisconnect
# matplotlib.use('TkAgg')

#Note that for ursina, all three element things like (a,b,c) means x=a, y=b, z=c
class Visualize:
    def __init__(self,grid_state,int_color,color_map,img_shape,plt):
        self.grid_state=grid_state #this would continuously update
        self.int_color=int_color
        self.color_map=color_map
        self.img_shape=img_shape
        self.plt=plt #this would also continuously updated
    
    #this function help visualize the grid
    def gridarray_to_image(self,grid_state,ThreeD=False,img_shape=None):
        if img_shape is None:
            img_shape=self.img_shape
            
        observation = np.zeros(img_shape,dtype=np.uint8)
        #Note: this is added in addition because we are now in another class instead of environment
        self.grid_state=grid_state
        """Note: if we are using this function in another class instead of inside the original environment class, we need to:
        1) Update grid_state based on updated situation
        """
        scale_x=int(observation.shape[0]/self.grid_state.shape[0])
        scale_y=int(observation.shape[1]/self.grid_state.shape[1])
        
        #color the map
        if(not ThreeD):
            for i in range(self.grid_state.shape[0]):
                for j in range(self.grid_state.shape[1]):
                    color_code=self.grid_state[i,j]
                    color_name=self.int_color[color_code]
                    pixal_value=self.color_map[color_name]
                        
                    observation[i*scale_x:(i+1)*scale_x,
                                j*scale_y:(j+1)*scale_y,:]=pixal_value
        else:
            return img_shape,scale_x,scale_y,observation
        
        return observation
    
    #Note that this function is currently static - we need to alter it if we want to show it with agent moving after adding the network
    def render(self,plt_updated,grid_state,mode="human",close=False,reward=None,count_time=0,delivered=0,ThreeD_vis=False,return64web=False,web_plt=None):
        #let's separate the webrender with the common one
        if(return64web):
            return self.web_render(grid_state,reward,count_time,delivered,web_plt)
        
        #close the window if we don't want to view the rendering animation
        if close:
            if hasattr(self,"fig"):
                self.plt.close(self.fig)
            return
        
        self.plt=plt_updated
        self.grid_state=grid_state
        if(not ThreeD_vis):
        #show the envrionment in rgb color
            img=self.gridarray_to_image(grid_state,ThreeD=ThreeD_vis,img_shape=None)
                
            #check if we have already started to view the warehouse animation
            if not hasattr(self,"fig"):
                (self.plt).ion()
                if(not return64web):
                    self.fig,self.ax=(self.plt).subplots(1,1)
                else:
                    self.fig,self.ax=(self.plt).subplots(1,1,figsize=(10,20))
                (self.plt).axis("off")
                (self.fig).canvas.manager.set_window_title("Warehouse Agent Navigation")
                (self.fig).tight_layout(pad=0)
                (self.ax).set_position([0,0,1,1])
                (self.fig).set_size_inches(12,4,forward=True)
                self.im=(self.ax).imshow(img)
                #legend reward
                self.reward_text=(self.fig).text(0.95,0.2,f"Total Reward:{reward:.2f}",ha='right',fontsize=12,color="chocolate")
                self.time_text=(self.fig).text(0.95,0.195,f"Total Time:{count_time:.1f} Sec",ha='right',fontsize=12,color="chocolate")
                self.delivered_text=(self.fig).text(0.95,0.19,f"Total Delivered:{delivered:d}",ha='right',fontsize=12,color="chocolate")

            else:
                #already very slow, not using time lag now
                # time.sleep(0.1)
                self.reward_text.set_text(f"Total Reward:{reward:.2f}")
                if(count_time<60):
                    self.time_text.set_text(f"Total Time:{count_time:.1f} Sec")
                elif(count_time>=60 and count_time<3600):
                    count_time=count_time/60
                    self.time_text.set_text(f"Total Time:{count_time:.1f} Min")
                else:
                    count_time=count_time/3600
                    self.time_text.set_text(f"Total Time:{count_time:.1f} Hrs")
                self.delivered_text.set_text(f"Total Delivered:{delivered:d}")
                (self.im).set_data(img)

            #since we are now using this for webapp, we put this line(plt.show) to the end & add "block=False"
            self.plt.show()
            (self.fig).canvas.draw()
            (self.fig).canvas.flush_events()
            
        else:
            image_shape,scl_x,scl_y,obs=self.gridarray_to_image(ThreeD=ThreeD_vis,img_shape=None)
            if not hasattr(self,"fig"):
                ThreeDV=ThreeD_visualization(self.grid_state,self.color_map,self.int_color,image_shape)
                self.ThreeDV=ThreeDV
                ThreeDV.pixal_value(self.cell_height,self.cell_width,scl_x,scl_y,obs)
                ThreeDV.rendering3D(reward,count_time)
                self.plt=ThreeDV.get_plt()
                self.fig=ThreeDV.get_fig()
                self.ax=ThreeDV.get_ax()
                self.ThreeDV=ThreeDV
            else:
                self.ThreeDV.pixal_value(self.cell_height,self.cell_width,scl_x,scl_y,obs)
                self.ThreeDV.rendering3D(reward,count_time)
                
    def web_render(self,grid_state,reward=None,count_time=0,delivered=0,web_plt=None):
            img = self.gridarray_to_image(grid_state, ThreeD=False, img_shape=None)

            fig, ax = plt.subplots(1, 1, figsize=(20, 6))
            ax.imshow(img)
            ax.axis("off")

            # Add stats
            fig.text(0.95, -0.1, f"Total Reward: {reward:.2f}",transform=ax.transAxes, ha='right', fontweight="bold", fontsize=14, color="black")
            fig.text(0.95, -0.17, f"Total Time: {count_time:.1f} Sec",transform=ax.transAxes, ha='right', fontweight="bold", fontsize=14, color="black")
            fig.text(0.95, -0.24, f"Total Delivered: {delivered}",transform=ax.transAxes, ha='right', fontweight="bold", fontsize=14, color="black")

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)  # Prevent memory leaks
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            return img_base64

    # #this is for 3Dlize
    # async def stream_layout(self,websocket: WebSocket, grid_state):
    #     await websocket.accept()
    #     try:
    #         layout = grid_state  # <- Make sure this returns a 2D grid of integers
    #         await websocket.send_json({"layout": layout})
    #         await asyncio.sleep(0.5)  # Control update rate
    #     except WebSocketDisconnect:
    #         print("Viewer disconnected")

class ThreeD_visualization:
    def __init__(self,grid_state,color_map,int_color,image_shape):
        self.grid_state=grid_state
        self.color_map=color_map
        self.int_color=int_color
        self.image_shape=image_shape
        self.visual=False
        self.fig=None
        self.plt=plt
    
    def pixal_value(self,cell_height,cell_width,scale_x,scale_y,observation):
        if(self.fig==None):
            fig=self.plt.figure()
            self.fig=fig
            ax=fig.add_subplot(cell_width,cell_height,1,projection='3d')
            self.ax=ax
        # fig=go.Figure()
        self.cell_height=cell_height
        self.cell_width=cell_width
        
        for i in range(self.grid_state.shape[0]):
            for j in range(self.grid_state.shape[1]):
                color_code=self.grid_state[i,j]
                color_name=self.int_color[color_code]
                rgb_color=self.color_map[color_name]
                #matplotlib 3D required
                rgb=tuple(pixal/255 for pixal in rgb_color) #notice that the colors need to be scaled to be able to be used in the later functions
                
                """
                note that:
                1) x,y,z: these are the coordinates
                2) the first fig action defined below are simply just points, not actual 3D things
                => only the update layout thing is actually turning the point into a 3D object
                """
                # width=i*self.cell_width
                # height=j*self.cell_height
                if(color_name=="gray"): #empty,height=0
                    self.ax.bar3d(x=i,y=j,z=0,dx=self.cell_width,dy=self.cell_height,dz=0,color=rgb)
                elif(color_name=="black"): #wall/shelves,height=15
                    self.ax.bar3d(x=i,y=j,z=0,dx=self.cell_width,dy=self.cell_height,dz=15,color=rgb)
                elif(color_name=="red"): #agent,height=5
                    self.ax.bar3d(x=i,y=j,z=0,dx=self.cell_width,dy=self.cell_height,dz=5,color=rgb)
                elif(color_name=="brown"): #package,height=3
                    self.ax.bar3d(x=i,y=j,z=0,dx=self.cell_width,dy=self.cell_height,dz=3,color=rgb)
                elif(color_name=="yellow"): #receiving region,height=0
                    self.ax.bar3d(x=i,y=j,z=0,dx=self.cell_width,dy=self.cell_height,dz=0,color=rgb)
                elif(color_name=="sky"): #transfer region,height=0
                    self.ax.bar3d(x=i,y=j,z=0,dx=self.cell_width,dy=self.cell_height,dz=0,color=rgb)
                else:
                    raise ValueError("The color is invalid here!")
                
                """Information about the Grid world if needed:
                1) 19 rows(can comprehend as 20 rows if needed)
                2) 70 columns(the x axis width)
                """
                observation[i*scale_x:(i+1)*scale_x,
                            j*scale_y:(j+1)*scale_y,
                            :]=rgb_color #now this stores the scaled rgb value for 3D

                self.visual=True
                #save to html(interactive 3D in browser)
                # fig.write_html("Warehouse_Agent_navigation.html")
        # return observation
        # return fig
    
    def get_plt(self):
        return self.plt
    
    def get_fig(self):
        return self.fig
    
    def get_ax(self):
        return self.ax
    
    def rendering3D(self,reward,count_time):
        if self.visual:
            # self.set_axes_equal(self.ax)
            height=self.grid_state.shape[1]*self.cell_height
            width=self.grid_state.shape[0]*self.cell_width
            depth=15
            (self.plt).ion()
            self.ax.set_box_aspect([width, height, depth]) #z-height: floor to ceiling
            self.ax.view_init(elev=30, azim=45)  # elevation and azimuth

            (self.fig).canvas.manager.set_window_title("Warehouse Agent Navigation 3D")
            self.reward_text=(self.fig).text(0.95,0.1,f"Total Reward:{reward:.2f}",ha='right',fontsize=12,color="chocolate")
            self.time_text=(self.fig).text(0.95,0.05,f"Total Time:{count_time:.1f} Sec",ha='right',fontsize=12,color="chocolate")
            self.visual=False #now in progress
            (self.fig).set_size_inches(12,4,forward=True)
            
            self.ax.set_xlim(0, width)
            self.ax.set_ylim(0, height)
            self.ax.set_zlim(0, depth)
            # self.fig.tight_layout()
        else:
            self.reward_text.set_text(f"Total Reward:{reward:.2f}")
            if(count_time<60):
                self.time_text.set_text(f"Total Time:{count_time:.1f} Sec")
            elif(count_time>=60 and count_time<3600):
                count_time=count_time/60
                self.time_text.set_text(f"Total Time:{count_time:.1f} Min")
            else:
                count_time=count_time/3600
                self.time_text.set_text(f"Total Time:{count_time:.1f} Hrs")
                
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.plt.show()
        (self.fig).canvas.draw()
        (self.fig).canvas.flush_events()
    
    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        
        max_range = max(x_range, y_range, z_range)

        x_middle = sum(x_limits) / 2
        y_middle = sum(y_limits) / 2
        z_middle = sum(z_limits) / 2

        ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
        ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
        ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

    """Important notes:
    x,y,z: list of 8 coordinates for cube corners(point of each corner)
    i,j,k: define which 3 points make each triangle face
    1) For x,y,z:
    point in term of which 3 faces intersect and form the point:
        Index 0(starting(x,y,z))(bottom-front-left), 1(bottom-front-right), 2(bottom-back-right), 3(bottom-back-left),
        4(top-front-left), 5(top-front-right), 6(top-back-right), 7(top-back-left)
    2) for each i,j,k at index i,
        these three points are the coordinate for the single triangle(2 triangles form a square)
        *since i,j,k are just referencing relative vertex indices, so even if we alter the cell size, it doesn't matter so much
    """
    def draw_grid(self,x,y,z,color_name,color_rgb):
        
        width=self.cell_width
        height=self.cell_height
        if(color_name=="gray"): #empty
            depth=0
        elif(color_name=="black"):#wall
            depth=15
        elif(color_name=="red"):#agent
            depth=5
        elif(color_name=="brown"):#package
            depth=3
        elif(color_name=="yellow"):#receiving
            depth=0
        elif(color_name=="sky"):#transfer
            depth=0
        y=y*depth
        
        #this face things from gpt
        faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [1, 2, 6, 5],  # right
        [3, 0, 4, 7]   # left
        ]
        
        return go.Mesh3d(
            x=[x,x+width,x+width,x,x,x+width,x+width,x],
            y=[y,y,y+depth,y+depth,y,y,y+depth,y+depth],
            z=[z,z,z,z,z+height,z+height,z+height,z+height],
            i=[face[0] for face in faces],
            j=[face[1] for face in faces],
            k=[face[2] for face in faces],
            color=color_rgb,
            opacity=0.9,
            flatshading=True,
            showscale=False
        )
        # return go.Bar3d(x=[x*width],y=[y],z=[z*height],dx=width,dy=depth,dz=height,color=color_rgb)