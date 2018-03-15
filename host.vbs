Dim oShell
Set oShell = WScript.CreateObject ("WScript.Shell")
oShell.run "jupyter notebook", 0
Set oShell = Nothing