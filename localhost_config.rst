Configuration of localhost
=============

You will need to have an alias called "localhost" that poitns to your SQL database.

1) Open SQL Server Configuration Manager
    - On the left pane, expand SQL Native Client 11.0 Configuration and right click Aliases. Select New Alias.
    - In the pop up dialog box, enter the following and then press OK:
        - Alias Name: localhost
        - Port No: 1433
        - Protocol: TCP/IP
        - Server: YOUR_COMPUTER_NAME/SQLEXPRESS (this should be something like HC2080)

2) On the left pane, expand SQL Server Network Configuration and click on Protocols for SQLEXPRESS.
    - Right click on TCP/IP and click Properties. In the dialog box, enter the following and then press OK:
        - Under the Protocol tab, verify that Enabled is set to Yes.
        - Under the IP Addresses tab, scroll all the way to the bottom.
        - Under IPALL, TCP Dynamic Ports should be blank
        - Under IPALL, TCP Port should be set to 1433

3)Open SSMS and verify that you can connect to the server localhost.
