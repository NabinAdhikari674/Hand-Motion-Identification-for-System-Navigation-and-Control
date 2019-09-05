
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
class Gui extends JFrame
{
  Robot r;

  public Gui()
      {
          createGUI();
      }

      private void createGUI()
      {

        JMenuBar mb = new JMenuBar();
        JMenu m1 = new JMenu("Key Control");
        JMenu m2 = new JMenu("Rebound ");
        mb.add(m1);
        mb.add(m2);
        JMenuItem m11 = new JMenuItem("WASD");
        JMenuItem m22 = new JMenuItem("ARROW");
        JMenuItem m33 = new JMenuItem("Mouse");
        JMenuItem m44 = new JMenuItem("Keyboard");
        m1.add(m11);
        m1.add(m22);
        m2.add(m33);
        m2.add(m44);
        setTitle("control");
        setSize(1000, 400);
        getContentPane().add(BorderLayout.NORTH, mb);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setVisible(true);

        try  { r=new Robot(); }
        catch (Exception e) {}

        addKeyListener(new KeyAdapter(){
            public void keyPressed(KeyEvent e)
            {
                if(r==null) return;
                Point p=MouseInfo.getPointerInfo().getLocation();
                switch(e.getKeyCode())
                {
                    case KeyEvent.VK_UP: r.mouseMove(p.x,p.y-4); break;
                    case KeyEvent.VK_DOWN: r.mouseMove(p.x,p.y+4); break;
                    case KeyEvent.VK_LEFT: r.mouseMove(p.x-4,p.y); break;
                    case KeyEvent.VK_RIGHT: r.mouseMove(p.x+4,p.y); break;
                    // left click
                    //case KeyEvent.VK_ENTER: r.mousePress(MouseEvent.BUTTON1_DOWN_MASK); r.mouseRelease(MouseEvent.BUTTON1_DOWN_MASK);
                }
            }

        });


      }

    public static void main(String args[])
     {
        new Gui();
    }
}
