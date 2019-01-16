import cmd

class CustomShell(cmd.Cmd):
  intro = 'Custom intro for shell'
  prompt = '(Custom) '

  def do_test(self, arg):
    'Test of the custom cmd'
    print(arg);

  def do_bye(self, arg):
    'Leave command'
    return True

CustomShell().cmdloop()
