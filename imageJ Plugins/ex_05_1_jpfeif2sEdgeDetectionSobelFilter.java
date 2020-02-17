import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.frame.*;

public class ex_05_1_jpfeif2sEdgeDetectionSobelFilter implements PlugIn {
//edge detection with sobelFilter
	public void run(String arg) {
		ImagePlus imp = IJ.getImage();
		boolean save = false; //toogle if all the images get saved or not
		
		//ex_05_1
		ImagePlus linesUp = sobelFilter(imp, "up",save);
		ImagePlus linesUp2 = sobelFilter(imp,"up2",save);
		ImagePlus linesSide = sobelFilter(imp,"side",save);
		ImagePlus linesSide2 = sobelFilter(imp, "side2",save);
		ImagePlus linesDiagonal = sobelFilter(imp,"diagonal",save);
		ImagePlus linesDiagonal2 = sobelFilter(imp, "diagonal2",save);
		
		ImagePlus[] lines =  {linesUp,linesUp2,linesSide,linesSide2};
 		ImagePlus[]lines2 = {linesDiagonal, linesDiagonal2};
 		
 		//display the sum of the filters
 		ImagePlus res = add(lines,"only_all_lines",save);
 		res = add(lines2, "all_diagonal_lines",save);
 		
 		
 		//ex_05_2
 		/*display the res of the filters as black lines over the original image.
 		 * like a comic
 		*/
 		res = sub(imp,linesUp,"linesUp",save);
 		res = sub(imp,linesUp2,"linesUp2",save);
 		res = sub(imp,linesSide,"linesSide",save);
 		res = sub(imp,linesSide2,"linesSide2",save);
 		res = sub(imp,linesDiagonal,"linesDiagonal",save);
 		res = sub(imp,linesDiagonal2,"linesDiagonal2",save);
 		res = sub(imp,lines2,"all_diagonal_lines",save);
 		res = sub(imp,lines,"all_Up_Side_lines",save);
		
	}
	//ex_05_1
	public ImagePlus sobelFilter(ImagePlus imp, String direction,boolean save) {
		ImagePlus res = null;
		if(direction.equals("up")) {
			int[][] structureUp = {{-1,0,1},{-2,0,2},{-1,0,1}};
			res = sobelFilter(imp,structureUp, direction);
		}else if(direction.equals("up2")) {
			int[][] structureUp2 = {{1,0,-1},{2,0,-2},{1,0,-1}};
			res = sobelFilter(imp,structureUp2, direction);
		}else if(direction.equals("side")) {
			int[][] structureSide = {{-1,-2,-1},{0,0,0},{+1,+2,+1}}; //[line][row]
			res = sobelFilter(imp,structureSide, direction);
		}else if(direction.equals("side2")){
			int[][] structureSide2 = {{1,2,1},{0,0,0},{-1,-2,-1}};
			res = sobelFilter(imp,structureSide2, direction);
		}else if(direction.equals("diagonal")) {
			int[][] structureDiagonal = {{-2,-1,0},{-1,0,1},{0,1,2}};
			res = sobelFilter(imp,structureDiagonal, direction);
		}else {
		int[][] structureDiagonal2 = {{2,1,0},{1,0,-1},{0,-1,-2}};
		res = sobelFilter(imp,structureDiagonal2, direction);
		}
		if(save) {
			IJ.save(res, res.getTitle());
		}
		return res;
	}
	
	//ex_05_1
	private ImagePlus sobelFilter(ImagePlus imp, int[][] structure,String direction) {
		//input is a grayscale image
		ImageProcessor ip = imp.getProcessor();
        
		int w = ip.getWidth();
		int h = ip.getHeight();
		ImagePlus resultImage = IJ.createImage("ex_05_1_SobelFilter_"+direction+".png", "8-Bit",w,h,1);
		ImageProcessor resultIp = resultImage.getProcessor();

		int size = structure.length;
		int[] c = new int[3]; 
		for(int u = 0; u<w;u++)
		{
			for(int v = 0; v<h;v++)
			 {
				
				//go through the mask
				int newVal = 0;
				for(int lineCounter = -(size/2); lineCounter <=(size/2); lineCounter++) {
					for(int rowCounter = -(size/2); rowCounter <=(size/2); rowCounter++) {
						c = ip.getPixel((u+rowCounter)%w,(v+lineCounter)%h,c);
						//found a pixel in the structure element
						newVal += structure[lineCounter+(size/2)][rowCounter+(size/2)]*c[0];
						
					}
				}
				
				resultIp.putPixel(u,v,(int)newVal);
			 }
		}
		resultImage.show();
		resultImage.draw();
		return resultImage;
	}
	
	//ex_05_1
	private ImagePlus add(ImagePlus[] lines, String designation,boolean save) {
		//imp-imp2 +offset
		//input are two Grayscale Images
		ImageProcessor[] linesIP = new ImageProcessor[lines.length];
		for(int i = 0; i<lines.length;i++) {
			linesIP[i] = lines[i].getProcessor();
		}
		
		int w = linesIP[0].getWidth();
		int h = linesIP[0].getHeight();
		//create blank 8-bit grayscale images in the correct size
		ImagePlus resImage = IJ.createImage("ex_5_1_Add_"+designation+".png", "8-Bit",w,h,1);
		ImageProcessor resIp = resImage.getProcessor();

		int[] cOrig = new int[3];
		int[] cLines = new int[3];
		int newVal=0;
		int cCombine =0;
		for(int u = 0; u<w;u++)
		{
			for(int v = 0; v<h;v++)
			 {
				cCombine = 0; 
				for(int i = 0; i<lines.length;i++) {
					cCombine += (linesIP[i].getPixel(u,v,cLines))[0];
				}
				resIp.putPixel(u,v,cCombine);
			}
		}
		//display images	
		resImage.show();
		resImage.draw();
		if(save) {
			IJ.save(resImage, resImage.getTitle());
		}
		return resImage;
	}
	
	
	
	//ex_05_2
	private ImagePlus sub(ImagePlus original, ImagePlus lines, String designation,boolean save) {
		ImagePlus[] l = {lines};
		return sub( original,  l, designation,save);
 	}
	//ex_05_2
	private ImagePlus sub(ImagePlus original, ImagePlus[] lines, String designation,boolean save) {
		//imp-imp2 +offset
		//input are two Grayscale Images
		ImageProcessor oIP = original.getProcessor();
		ImageProcessor[] linesIP = new ImageProcessor[lines.length];
		for(int i = 0; i<lines.length;i++) {
			linesIP[i] = lines[i].getProcessor();
		}
		
		int w = oIP.getWidth();
		int h = oIP.getHeight();
		//create blank 8-bit grayscale images in the correct size
		ImagePlus resImage = IJ.createImage("ex_5_1_Sub_"+designation+".png", "8-Bit",w,h,1);
		ImageProcessor resIp = resImage.getProcessor();

		int[] cOrig = new int[3];
		int[] cLines = new int[3];
		int newVal=0;
		int cCombine =0;
		for(int u = 0; u<w;u++)
		{
			for(int v = 0; v<h;v++)
			 {
				cCombine = 0; 
				for(int i = 0; i<lines.length;i++) {
					cCombine += (linesIP[i].getPixel(u,v,cLines))[0];
				}
				cOrig = oIP.getPixel(u,v,cOrig);
				newVal = cOrig[0]-cCombine;
				resIp.putPixel(u,v,newVal);
			}
		}
		//display images	
		resImage.show();
		resImage.draw();
		if(save) {
			IJ.save(resImage, resImage.getTitle());
		}
		return resImage;
	}

	
}
