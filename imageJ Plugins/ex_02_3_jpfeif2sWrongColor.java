import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.frame.*;

public class ex_02_3_jpfeif2sWrongColor implements PlugIn {
//grayscale to wrong color 
	public void run(String arg) {
		ImagePlus imp = IJ.getImage();
		wrongcolor(imp.duplicate());
		
		
	}
	private void wrongcolor (ImagePlus imp){
		ImageProcessor ip = imp.getProcessor();

		int w = ip.getWidth();
		int h = ip.getHeight();
		
		//create blank 8-bit grayscale images in the correct size
		ImagePlus grayImage = IJ.createImage("ex_02_3_Nachher (Falschfarbenbild)", "RGB-24",w,h,1);
		ImageProcessor gIp = grayImage.getProcessor();

		int[] c = new int[3]; 
		for(int u = 0; u<w;u++)
		{
			for(int v = 0; v<h;v++)
			 {
				//get colors
				c = ip.getPixel(u,v,c);

				//input is a grayscale image.
				int g = c[0];
				//24 bits is 8 8 8 for RGB. no alpha channel
				
				//new = blow +(bhigh-blow/ahigh-alow)*(i-alow)
				int red = (int) 255+((((120-255)/(65-0)))*(g-0));
				int green=(int) (120+((((255-120)/(127-100))) * (g-100))) ;
				int green2 =(int) (255+((((120-255)/(170-127))) * (g-127)));
				int blue=(int) 120+((((255-120)/(255-200)))*(g-200));
				red = clip(red);
				green = gclip(clip(green)+clip(green2));
				blue = clip(blue);
				
				int [] rgb = {red,green,blue}; 
				
				gIp.putPixel(u,v,rgb);
				
			}
		}
		//display images	
		grayImage.show();
		grayImage.draw();
		}
	private int gclip(int a) {
		if(a>255)
			return 255;
		return a;		
	}
	private int clip (int a) {
		if(a<0)
			return 0;
		if(a>255)
			return 0;
		return a;
	}

}
